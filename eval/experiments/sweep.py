"""
Thorough isolated sweep of SLMs / NLI encoders vs the production yes/no task.
Does not modify or restart the production 8B service.

Writes one JSON line per model to sweep_results.jsonl as it goes.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.expanduser("~/tab_complete_poc_bench"))
sys.path.insert(0, os.path.expanduser("~/tab_complete_poc_bench/eval_experiments"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "classifier_service"))

from test_cases import CASES  # noqa: E402
from test_cases_holdout import CASES as HOLDOUT  # noqa: E402

try:
    from rules import rule_types, split_context
except Exception:
    sys.path.insert(0, os.path.expanduser("~/tab_complete_poc_bench"))
    from rules import rule_types, split_context  # noqa: E402

from router import QUESTIONS, TYPES, build_prefix, LMYesNo  # noqa: E402

OUT = os.environ.get("SWEEP_OUT", os.path.join(os.path.dirname(__file__), "sweep_results.jsonl"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log(msg):
    print(msg, flush=True)


class SequentialYesNo:
    """Same verbalizer as LMYesNo, but one type at a time (for hybrid-attn models)."""

    def __init__(self, model_name: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, trust_remote_code=True, device_map="cuda"
        ).eval()
        self.device = next(self.model.parameters()).device
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.prefix_ids = self.tok(build_prefix(), return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            self.cache = self.model(self.prefix_ids, use_cache=True).past_key_values
        self.yes = self.tok.encode(" yes", add_special_tokens=False)[0]
        self.no = self.tok.encode(" no", add_special_tokens=False)[0]
        self.plen = self.prefix_ids.shape[1]

    @torch.no_grad()
    def probs(self, text: str) -> dict[str, float]:
        out = {}
        for t in TYPES:
            ids = self.tok.encode(f'Text: "{text}"\n{QUESTIONS[t]}', add_special_tokens=False)
            input_ids = torch.tensor([ids], device=self.device)
            n = len(ids)
            attn = torch.ones(1, self.plen + n, device=self.device, dtype=torch.long)
            pos = (self.plen + torch.arange(n, device=self.device)).unsqueeze(0)
            cache = copy.deepcopy(self.cache)
            logits = self.model(
                input_ids=input_ids, attention_mask=attn, position_ids=pos, past_key_values=cache
            ).logits
            yn = logits[0, n - 1, [self.yes, self.no]].float()
            out[t] = torch.softmax(yn, -1)[0].item()
        return out


class NLINext:
    def __init__(self, model_name: str, template: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
        self.entail = [i for i, l in self.model.config.id2label.items() if "entail" in l.lower()]
        self.entail = self.entail[0] if self.entail else 0
        descs = {
            "person": "a salesperson's name",
            "location": "a city or geographic location",
            "product": "a product name",
            "client": "a client or company name",
        }
        self.hyps = [template.format(descs[t]) for t in TYPES]
        self.labels = TYPES

    @torch.no_grad()
    def probs(self, text: str) -> dict[str, float]:
        inputs = self.tok(
            [text] * len(self.hyps), self.hyps, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        scores = torch.softmax(self.model(**inputs).logits, -1)[:, self.entail].tolist()
        return dict(zip(self.labels, scores))


def decide(scores, th):
    return {t for t in TYPES if scores.get(t, 0) >= th}


def eval_scores(cases, get_scores, th, use_rules):
    ok, times = 0, []
    get_scores(cases[0][0])
    for text, req, opt in cases:
        t0 = time.time()
        if use_rules:
            rule = rule_types(split_context(text))
            pred = set(rule) if rule else decide(get_scores(text), th)
        else:
            pred = decide(get_scores(text), th)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
        if req <= pred <= (req | opt):
            ok += 1
    times.sort()
    return {
        "ok": ok,
        "n": len(cases),
        "acc": ok / len(cases),
        "median_ms": times[len(times) // 2],
        "p90_ms": times[int(len(times) * 0.9)],
    }


def write_result(row):
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    log(json.dumps(row))


def load_yesno(model_name):
    try:
        lm = LMYesNo(model_name)
        lm.probs("warmup")
        return lm, "batched"
    except Exception as e:
        log(f"batched failed for {model_name}: {e}; trying sequential")
        torch.cuda.empty_cache()
        lm = SequentialYesNo(model_name)
        lm.probs("warmup")
        return lm, "sequential"


def run_yesno(model_name, extra=None):
    extra = extra or {}
    log(f"\n===== YESNO {model_name} =====")
    lm, mode = load_yesno(model_name)
    row = {"kind": "yesno", "model": model_name, "mode": mode, **extra}
    for split, cases in (("main", CASES), ("holdout", HOLDOUT)):
        for th in (0.5, 0.6):
            r = eval_scores(cases, lm.probs, th, use_rules=False)
            row[f"{split}_th{th}"] = r
            log(f"  {split} th={th}: {r['ok']}/{r['n']} = {r['acc']:.1%}  med {r['median_ms']:.1f}ms")
        r = eval_scores(cases, lm.probs, 0.6, use_rules=True)
        row[f"{split}_rules"] = r
        log(f"  {split} +rules: {r['ok']}/{r['n']} = {r['acc']:.1%}")
    del lm
    torch.cuda.empty_cache()
    write_result(row)
    return row


def run_nli(model_name, template):
    log(f"\n===== NLI {model_name} =====")
    clf = NLINext(model_name, template)
    clf.probs("warmup")
    row = {"kind": "nli", "model": model_name, "template": template}
    for split, cases in (("main", CASES), ("holdout", HOLDOUT)):
        for th in (0.3, 0.5, 0.6):
            r = eval_scores(cases, clf.probs, th, use_rules=False)
            row[f"{split}_th{th}"] = r
            log(f"  {split} th={th}: {r['ok']}/{r['n']} = {r['acc']:.1%}  med {r['median_ms']:.1f}ms")
        r = eval_scores(cases, clf.probs, 0.5, use_rules=True)
        row[f"{split}_rules"] = r
        log(f"  {split} +rules: {r['ok']}/{r['n']} = {r['acc']:.1%}")
    del clf
    torch.cuda.empty_cache()
    write_result(row)
    return row


def already_done(model_name):
    if not os.path.exists(OUT):
        return False
    for line in open(OUT):
        try:
            if json.loads(line).get("model") == model_name:
                return True
        except Exception:
            pass
    return False


# Research shortlist (Oct 2026): SLMs that reviews claim can match or
# approach Qwen3-8B at similar/smaller size, plus the remaining standard
# NLP zero-shot encoders. Cached models first so the sweep is not blocked
# by Hub downloads. See RESULTS.md for sources and scores.
#
# Skipped on purpose (already measured, or same failure class / larger):
#   Laya / Clef-flash          System-1 aboutness, 9–18%
#   spaCy NER / POS            wrong task / already in rules.py
#   GLiNER / GLiClass          mentioned-entity / topic, not next-slot
#   MiniLM / BGE cosine        topical overlap, already failed
#   DeBERTa-v3-xsmall NLI      aboutness, already failed
#   DeBERTa-v3-base NLI        26% model-only (this sweep)
#   ModernBERT-large NLI       same NLI template; Hub download hung
#   DeBERTa-v3-large NLI       same NLI template; no reason to expect
#                               next-slot if base is 26%
#   Qwen2.5-0.5B JSON          generation already failed
#   Clef-flash 9B              larger than 8B
JOBS = [
    # --- already on the GPU box ---
    ("yesno", "Qwen/Qwen3-4B-Instruct-2507"),
    ("yesno", "Qwen/Qwen3.5-0.8B"),
    ("yesno", "google/gemma-3-1b-it"),
    ("yesno", "Qwen/Qwen3-1.7B-Base"),
    ("yesno", "Qwen/Qwen3.5-2B"),
    ("yesno", "Qwen/Qwen3-8B"),
    # --- 3–4B class claimed to rival 7–9B ---
    ("yesno", "Qwen/Qwen3.5-4B"),
    ("yesno", "microsoft/Phi-4-mini-instruct"),
    ("yesno", "google/gemma-3-4b-it"),
    ("yesno", "HuggingFaceTB/SmolLM3-3B"),
    ("yesno", "ibm-granite/granite-4.1-3b"),
    ("yesno", "openbmb/MiniCPM5-2B"),
    ("yesno", "Qwen/Qwen2.5-3B-Instruct"),
    ("yesno", "LiquidAI/LFM2.5-2.6B"),
    ("yesno", "meta-llama/Llama-3.2-3B-Instruct"),
    ("yesno", "mistralai/Ministral-3-3B-Instruct-2512"),
    # --- sub-2B that reviews still list as useful ---
    ("yesno", "LiquidAI/LFM2.5-1.2B-Instruct"),
    ("yesno", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("yesno", "Qwen/Qwen2.5-1.5B-Instruct"),
    # --- similar size to current 8B ---
    ("yesno", "Qwen/Qwen2.5-7B-Instruct"),
    ("yesno", "allenai/Olmo-3-7B-Instruct"),
    # --- remaining NLP zero-shot encoder (classic MNLI) ---
    ("nli", "facebook/bart-large-mnli", "The next words the user types will be {}."),
]


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    log(f"device={DEVICE} out={OUT}")
    for job in JOBS:
        kind, name = job[0], job[1]
        if only and name not in only and not any(name.endswith(x) or x in name for x in only):
            continue
        if already_done(name):
            log(f"skip already done: {name}")
            continue
        try:
            if kind == "nli":
                run_nli(name, job[2])
            else:
                run_yesno(name)
        except Exception:
            log(f"FAILED {name}\n{traceback.format_exc()}")
            write_result({"kind": kind, "model": name, "error": traceback.format_exc()[-500:]})
            torch.cuda.empty_cache()
    log("SWEEP DONE")


if __name__ == "__main__":
    main()
