"""
Next-entity router: given the text typed so far, which entity type(s) could
the user type NEXT?

Two layers, no training anywhere:
  1. rules.explicit_type -- the user literally named the type right before the
     slot ("for the client", "in the city of"). Deterministic.
  2. A causal LLM (Qwen3-8B) scored few-shot: for each type, P(yes) to
     "Could the next words be a <type>?". All 4 questions run as ONE batched
     forward pass on top of a KV-cached few-shot prefix, giving independent
     (multi-label) probabilities.

See APPROACH.md for the full history (rejected models, why yes/no scoring
rather than generation, how the KV-cached verbalizer stays ~130ms).
Eval: eval/eval_service.py, eval/test_cases.py, eval/test_cases_holdout.py.
"""
import copy
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rules import rule_types, split_context

TYPES = ["person", "location", "product", "client"]
P, L, R, C = TYPES
ALL = set(TYPES)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_THRESHOLD = 0.6

HEADER = (
    "A user is typing a question into a sales analytics search box. For each partial "
    "question, decide what kind of value the user could type NEXT, right after the text. "
    "The last word may be a partially typed value. Values can be a salesperson's name, a "
    "location (city or region), a product name, or a client company.\n\n"
)
QUESTIONS = {
    "person": "Could the next words be a salesperson's name?",
    "location": "Could the next words be a location?",
    "product": "Could the next words be a product name?",
    "client": "Could the next words be a client company?",
}
FEW_SHOT = [
    ("Get me the revenue numbers for", ALL),
    ("Display the figures for mar", ALL),
    ("Revenue for the salesperson", {P}),
    ("Quarterly numbers for the customer", {C}),
    ("Revenue generated from the item", {R}),
    ("Deals in the region of", {L}),
    ("Orders from accounts located in", {L}),
    ("What quantity of", {R}),
    ("Bonus paid to", {P}),
    ("Which rep closed the most deals with", {C}),
    ("In Mumbai, how much revenue came from", {P, R, C}),
    ("Units of Smart Desk Lamp shipped to", {L, C}),
    ("Benchmark Toronto against", {L}),
    ("Benchmark Globex Inc against", {C}),
    ("Compare Paris and", {L}),
    ("Compare Acme Corp and", {C}),
    ("Compare Priya Sharma and", {P}),
    ("Compare iPhone 15 and", {R}),
    ("Which item had the best margin in", {L}),
    ("Which city ordered the most of", {R}),
    ("Best selling items for", {P, L, C}),
    ("Show me the overall profit margin", set()),
    ("How is the business trending", set()),
    # Generic grouping / "for" should stay fully open -- the model otherwise
    # drops person or client on "split by" / "growth rate for".
    ("Break the totals down by", ALL),
    ("What's the change in revenue for", ALL),
]


def build_prefix(seed: int = 0) -> str:
    rng = random.Random(seed)
    blocks = []
    for text, types in FEW_SHOT:
        order = TYPES[:]
        rng.shuffle(order)  # random order so any question may directly follow "Text:"
        lines = [f'Text: "{text}"'] + [f"{QUESTIONS[t]} {'yes' if t in types else 'no'}" for t in order]
        blocks.append("\n".join(lines))
    return HEADER + "\n\n".join(blocks) + "\n\n"


class LMYesNo:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(self.device).eval()
        self.prefix_ids = self.tok(build_prefix(), return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            self.cache = self.model(self.prefix_ids, use_cache=True).past_key_values
        self.yes = self.tok.encode(" yes", add_special_tokens=False)[0]
        self.no = self.tok.encode(" no", add_special_tokens=False)[0]

    @torch.no_grad()
    def probs(self, text: str) -> dict[str, float]:
        enc = [self.tok.encode(f'Text: "{text}"\n{QUESTIONS[t]}', add_special_tokens=False) for t in TYPES]
        lens = [len(e) for e in enc]
        maxlen = max(lens)
        pad = self.tok.pad_token_id or 0
        dev = self.device
        ids = torch.tensor([e + [pad] * (maxlen - len(e)) for e in enc], device=dev)
        plen = self.prefix_ids.shape[1]
        attn = torch.cat(
            [torch.ones(len(TYPES), plen, device=dev, dtype=torch.long),
             torch.tensor([[1] * n + [0] * (maxlen - n) for n in lens], device=dev)], dim=1)
        pos = (plen + torch.arange(maxlen, device=dev)).unsqueeze(0).expand(len(TYPES), -1)
        cache = copy.deepcopy(self.cache)
        cache.batch_repeat_interleave(len(TYPES))
        logits = self.model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache).logits
        out = {}
        for i, t in enumerate(TYPES):
            yn = logits[i, lens[i] - 1, [self.yes, self.no]].float()
            out[t] = torch.softmax(yn, -1)[0].item()
        return out


class Router:
    def __init__(self, model_name: str = DEFAULT_MODEL, threshold: float = DEFAULT_THRESHOLD, use_rules: bool = True):
        self.lm = LMYesNo(model_name)
        self.threshold = threshold
        self.use_rules = use_rules
        self.lm.probs("warmup")

    def route(self, text: str) -> dict:
        rule = rule_types(split_context(text)) if self.use_rules else None
        if rule:
            scores = {t: float(t in rule) for t in TYPES}
            source = "rule"
            entity_types = [t for t in TYPES if t in rule]
        else:
            scores = self.lm.probs(text)
            source = "llm"
            entity_types = [t for t in TYPES if scores[t] >= self.threshold]
        return {"entity_types": entity_types, "scores": scores, "source": source}
