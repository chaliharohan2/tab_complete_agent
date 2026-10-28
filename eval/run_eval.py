"""
Evaluate next-entity routing approaches against eval/test_cases.py.

Usage (on the GPU box):
    python run_eval.py nli_about
    python run_eval.py nli_next
    python run_eval.py lm_dist Qwen/Qwen2.5-1.5B-Instruct
    python run_eval.py lm_gen  Qwen/Qwen2.5-1.5B-Instruct
"""
import os
import sys
import time

import torch

from test_cases import ALL, NONE, C, L, P, R

if os.environ.get("TESTSET") == "holdout":
    from test_cases_holdout import CASES
else:
    from test_cases import CASES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TYPES = ["person", "location", "product", "client"]

# --------------------------------------------------------------------------
# NLI zero-shot (current approach and a next-slot hypothesis variant)
# --------------------------------------------------------------------------
NLI_MODEL = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
NLI_DESCS = {
    "person": "a sales person or salesperson's name",
    "location": "a city, place, or geographic location",
    "product": "a product or item name",
    "client": "a client, customer, or company name",
}
NLI_OFF_TOPIC = (
    "a general question that is not asking to search for a specific "
    "person, location, product, or client"
)


class NLIClassifier:
    def __init__(self, template: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(DEVICE).eval()
        id2label = self.model.config.id2label
        self.entail = [i for i, l in id2label.items() if "entail" in l.lower()][0]
        self.labels = TYPES + ["off_topic"]
        self.hyps = [template.format(NLI_DESCS[t]) for t in TYPES] + [template.format(NLI_OFF_TOPIC)]

    @torch.no_grad()
    def classify(self, text):
        inputs = self.tok([text] * len(self.hyps), self.hyps, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        scores = torch.softmax(self.model(**inputs).logits, -1)[:, self.entail].tolist()
        s = dict(zip(self.labels, scores))
        ent = {t: s[t] for t in TYPES}
        top_t, top = max(ent.items(), key=lambda kv: kv[1])
        if s["off_topic"] > top + 0.15:
            return set()
        if top < 0.20:
            return set(ALL)
        return {t for t, v in ent.items() if v >= 0.6 * top}


# --------------------------------------------------------------------------
# Causal LM, few-shot, framed as "what kind of value comes NEXT"
# --------------------------------------------------------------------------
LM_PROMPT = """A user is typing a question into a sales analytics search box. Predict what kind of entity value the user will type NEXT, right after the text so far. The last word may be a partially typed value.
Entity kinds: person (a salesperson), location (a city or region), product, client (a customer company).
Answer "all" if any kind could come next, the specific kind(s) if the wording narrows it down, or "none" if no entity value is expected next.

Text: Get me the revenue numbers for
Next: all
Text: Get me the revenue numbers for mar
Next: all
Text: Revenue for the salesperson
Next: person
Text: Quarterly numbers for the customer
Next: client
Text: Revenue generated from the item
Next: product
Text: Deals in the region of
Next: location
Text: Orders from customers based in
Next: location
Text: What quantity of
Next: product
Text: Revenue brought in by
Next: person, client
Text: Which rep closed the most deals with
Next: client
Text: In Mumbai, how much revenue came from
Next: person, product, client
Text: Units of Smart Desk Lamp shipped to
Next: location, client
Text: Contrast Toronto with
Next: location
Text: Show me the overall profit margin
Next: none
Text: {text}
Next:"""

LM_LABELS = TYPES + ["all", "none"]


class LMClassifier:
    def __init__(self, model_name: str, mode: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.mode = mode
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(DEVICE).eval()
        self.label_ids = {}
        for l in LM_LABELS:
            ids = self.tok.encode(" " + l, add_special_tokens=False)
            self.label_ids[l] = ids[0]
        assert len(set(self.label_ids.values())) == len(LM_LABELS), self.label_ids

    def _inputs(self, text):
        return self.tok(LM_PROMPT.format(text=text), return_tensors="pt").to(DEVICE)

    @torch.no_grad()
    def classify(self, text):
        inputs = self._inputs(text)
        if self.mode == "dist":
            logits = self.model(**inputs).logits[0, -1]
            ids = torch.tensor([self.label_ids[l] for l in LM_LABELS], device=logits.device)
            probs = dict(zip(LM_LABELS, torch.softmax(logits[ids].float(), -1).tolist()))
            top = max(probs, key=probs.get)
            if top == "none":
                return set()
            if top == "all":
                return set(ALL)
            p_top = probs[top]
            return {t for t in TYPES if probs[t] >= 0.3 * p_top}

        out = self.model.generate(
            **inputs, max_new_tokens=12, do_sample=False, pad_token_id=self.tok.eos_token_id
        )
        gen = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        line = gen.strip().split("\n")[0]
        parts = {p.strip().lower() for p in line.split(",")}
        if "none" in parts:
            return set()
        if "all" in parts:
            return set(ALL)
        return {p for p in parts if p in TYPES}


# --------------------------------------------------------------------------
# Causal LM, value likelihood: how probable are example VALUES of each type as
# the literal continuation of the text? No prompt, no instructions.
# --------------------------------------------------------------------------
EXEMPLARS = {
    "person": ["John Smith", "Priya Sharma", "Michael Brown", "Sarah Johnson", "Rahul Verma", "Emily Chen"],
    "location": ["Mumbai", "London", "New York", "Texas", "Singapore", "Berlin"],
    "product": ["iPhone 15", "MacBook Pro", "Galaxy S24", "ThinkPad X1", "AirPods Pro", "Dell XPS 13"],
    "client": ["Acme Corp", "Globex Inc", "Tata Motors", "Walmart", "Infosys", "Reliance Industries"],
}

from rules import split_context  # noqa: E402


class LMValueScorer:
    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(DEVICE).eval()
        self.ex = [(t, self.tok.encode(" " + v, add_special_tokens=False)) for t in TYPES for v in EXEMPLARS[t]]

    @torch.no_grad()
    def type_scores(self, context: str) -> dict:
        prefix = self.tok.encode(context, add_special_tokens=False)
        seqs = [prefix + ids for _, ids in self.ex]
        maxlen = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        input_ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs], device=DEVICE)
        attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs], device=DEVICE)
        logp = torch.log_softmax(self.model(input_ids=input_ids, attention_mask=attn).logits.float(), -1)
        per_ex = []
        for i, (_, ids) in enumerate(self.ex):
            start = len(prefix)
            positions = torch.arange(start - 1, start - 1 + len(ids), device=DEVICE)
            per_ex.append(logp[i, positions, torch.tensor(ids, device=DEVICE)].sum())
        per_ex = torch.stack(per_ex)
        out = {}
        for t in TYPES:
            idx = [i for i, (tt, _) in enumerate(self.ex) if tt == t]
            out[t] = (torch.logsumexp(per_ex[idx], 0) - torch.log(torch.tensor(float(len(idx))))).item()
        return out


def decide_from_values(scores: dict, ratio: float, none_floor: float) -> set:
    if max(scores.values()) < none_floor:
        return set()
    t = torch.tensor([scores[k] for k in TYPES])
    probs = dict(zip(TYPES, torch.softmax(t, 0).tolist()))
    top = max(probs.values())
    return {k for k, p in probs.items() if p >= ratio * top}


CALIBRATION_CONTEXTS = ["N/A", "The answer is", "Name:", "Here is one example:"]


def eval_values(model_name: str):
    clf = LMValueScorer(model_name)
    clf.type_scores("warmup")
    calib = {t: 0.0 for t in TYPES}
    if os.environ.get("CALIBRATE"):
        for c in CALIBRATION_CONTEXTS:
            for t, v in clf.type_scores(c).items():
                calib[t] += v / len(CALIBRATION_CONTEXTS)
    rows, times = [], []
    for text, req, opt in CASES:
        ctx = split_context(text)
        t0 = time.time()
        s = clf.type_scores(ctx) if ctx else {t: 0.0 for t in TYPES}
        s = {t: v - calib[t] for t, v in s.items()}
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        rows.append((text, ctx, s, req, opt))
    times.sort()
    print(f"### lm_values {model_name}")
    print(f"latency: median {times[len(times) // 2]:.1f}ms, p90 {times[int(len(times) * 0.9)]:.1f}ms")

    none_best = sorted(max(s.values()) for _, _, s, req, opt in rows if not req and not opt)
    ent_best = sorted(max(s.values()) for _, _, s, req, opt in rows if req)
    print(f"best-type logprob  none cases: {[round(x, 1) for x in none_best]}")
    print(f"best-type logprob  entity cases (lowest 5): {[round(x, 1) for x in ent_best[:5]]}")

    grid = []
    for ratio in (0.15, 0.25, 0.35, 0.5):
        for floor in (-99.0, -16.0, -14.0, -12.0, -10.0):
            ok = sum(req <= decide_from_values(s, ratio, floor) <= (req | opt) for _, _, s, req, opt in rows)
            grid.append((ok, ratio, floor))
    for ok, ratio, floor in sorted(grid, reverse=True)[:6]:
        print(f"  ratio={ratio} none_floor={floor}: {ok}/{len(rows)} = {ok / len(rows):.1%}")

    ratio, floor = float(os.environ.get("RATIO", 0.25)), float(os.environ.get("FLOOR", -99))
    ok = 0
    for text, ctx, s, req, opt in rows:
        pred = decide_from_values(s, ratio, floor)
        if req <= pred <= (req | opt):
            ok += 1
        elif not os.environ.get("QUIET"):
            sc = " ".join(f"{k[:3]}={v:.1f}" for k, v in s.items())
            exp = fmt(req) + (f" (+opt {fmt(opt)})" if opt else "")
            print(f"  FAIL {text!r:55s} got {fmt(pred):32s} exp {exp:40s} [{sc}]")
    print(f"at ratio={ratio} floor={floor}: {ok}/{len(rows)} = {ok / len(rows):.1%}\n")


def eval_yesno(model_name: str):
    # Uses the exact production router (classifier_service/router.py).
    from router import Router

    hybrid = bool(os.environ.get("HYBRID"))
    router = Router(model_name, use_rules=hybrid)
    rows, times = [], []
    for text, req, opt in CASES:
        t0 = time.time()
        p = router.route(text)["scores"]
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        rows.append((text, p, req, opt))
    times.sort()
    print(f"### lm_yesno{' +rules' if hybrid else ''} {model_name}  (testset={os.environ.get('TESTSET', 'main')})")
    print(f"latency: median {times[len(times) // 2]:.1f}ms, p90 {times[int(len(times) * 0.9)]:.1f}ms")
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        ok = sum(req <= {t for t, v in p.items() if v >= th} <= (req | opt) for _, p, req, opt in rows)
        print(f"  threshold={th}: {ok}/{len(rows)} = {ok / len(rows):.1%}")
    th = float(os.environ.get("TH", 0.5))
    if not os.environ.get("QUIET"):
        for text, p, req, opt in rows:
            pred = {t for t, v in p.items() if v >= th}
            if not (req <= pred <= (req | opt)):
                sc = " ".join(f"{k[:3]}={v:.2f}" for k, v in p.items())
                exp = fmt(req) + (f" (+opt {fmt(opt)})" if opt else "")
                print(f"  FAIL {text!r:55s} got {fmt(pred):32s} exp {exp:40s} [{sc}]")
    print()


def fmt(s):
    return "{" + ",".join(sorted(s)) + "}" if s else "{}"


def main():
    backend = sys.argv[1]
    if backend == "lm_values":
        return eval_values(sys.argv[2])
    if backend == "lm_yesno":
        return eval_yesno(sys.argv[2])
    if backend == "nli_about":
        clf = NLIClassifier("This example is about {}.")
    elif backend == "nli_next":
        clf = NLIClassifier("The next words the user types will be {}.")
    elif backend in ("lm_dist", "lm_gen"):
        clf = LMClassifier(sys.argv[2], backend.split("_")[1])
    else:
        raise SystemExit(f"unknown backend {backend}")

    clf.classify("warmup")
    correct, times, failures = 0, [], []
    for text, req, opt in CASES:
        t0 = time.time()
        pred = clf.classify(text)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        if req <= pred <= (req | opt):
            correct += 1
        else:
            failures.append((text, pred, req, opt))

    name = backend + (f" {sys.argv[2]}" if len(sys.argv) > 2 else "")
    times.sort()
    print(f"### {name}")
    print(f"accuracy: {correct}/{len(CASES)} = {correct / len(CASES):.1%}")
    print(f"latency: median {times[len(times) // 2]:.1f}ms, p90 {times[int(len(times) * 0.9)]:.1f}ms")
    for text, pred, req, opt in ([] if os.environ.get("QUIET") else failures):
        exp = fmt(req) + (f" (+optional {fmt(opt)})" if opt else "")
        print(f"  FAIL {text!r:58s} got {fmt(pred):35s} expected {exp}")
    print()


if __name__ == "__main__":
    main()
