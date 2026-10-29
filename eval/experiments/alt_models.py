"""
Isolated bake-off for alternatives to Qwen3-8B. Does NOT import or modify
the production classifier service process.

Backends:
  laya_choice   -- convaiinnovations/laya, one multi-option choice question
  laya_noul     -- same model, 4 independent yes/no (noul) questions
  gliner        -- urchade/gliner_small-v2.1 zero-shot NER (mentioned entities)
  qwen_yesno    -- same verbalizer as production, different model name

Usage (on the GPU box):
    python alt_models.py laya_choice
    python alt_models.py laya_noul
    python alt_models.py gliner
    python alt_models.py qwen_yesno Qwen/Qwen3-4B
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from test_cases import ALL, CASES, C, L, P, R  # noqa: E402
from test_cases_holdout import CASES as HOLDOUT  # noqa: E402

TYPES = ["person", "location", "product", "client"]


def fmt(s):
    return "{" + ",".join(sorted(s)) + "}" if s else "{}"


def score(cases, predict, name, quiet=False):
    ok, times, fails = 0, [], []
    predict(cases[0][0])  # warmup
    for text, req, opt in cases:
        t0 = time.time()
        pred = predict(text)
        times.append((time.time() - t0) * 1000)
        if req <= pred <= (req | opt):
            ok += 1
        else:
            fails.append((text, pred, req, opt))
    times.sort()
    print(f"### {name}: {ok}/{len(cases)} = {ok / len(cases):.1%}   "
          f"median {times[len(times) // 2]:.1f}ms  p90 {times[int(len(times) * 0.9)]:.1f}ms")
    if not quiet:
        for text, pred, req, opt in fails:
            exp = fmt(req) + (f" (+opt {fmt(opt)})" if opt else "")
            print(f"  FAIL {text!r:55s} got {fmt(pred):32s} exp {exp}")
    print()
    return ok, len(cases)


# --------------------------------------------------------------------------
# Laya
# --------------------------------------------------------------------------
LAYA_CHOICE = {
    "next": {
        "type": "choice",
        "instructions": (
            "A user is typing a sales-analytics question. What kind of entity "
            "value will they type NEXT, right after this text? The last word "
            "may be a partially typed value."
        ),
        "criteria": {
            "person": "a salesperson's name comes next",
            "location": "a city or geographic location comes next",
            "product": "a product name comes next",
            "client": "a client or company name comes next",
            "all": "any of person, location, product, or client could come next",
            "none": "no entity value is expected next (the question is complete or not asking for one)",
        },
    }
}

LAYA_NOUL = {
    "person": {"type": "noul", "instructions": "Could the next words the user types be a salesperson's name?"},
    "location": {"type": "noul", "instructions": "Could the next words the user types be a city or location?"},
    "product": {"type": "noul", "instructions": "Could the next words the user types be a product name?"},
    "client": {"type": "noul", "instructions": "Could the next words the user types be a client or company name?"},
}


def load_laya(subfolder=None):
    import laya

    os.environ.setdefault("USE_TF", "0")
    if subfolder:
        return laya.load("convaiinnovations/laya", subfolder=subfolder)
    return laya.load("convaiinnovations/laya")


def make_laya_choice(agent, threshold=0.25):
    def predict(text):
        res = agent.predict(text, LAYA_CHOICE)
        ans = res["answers"]["next"]
        choice = ans.get("choice")
        # Some versions put per-option probs under "probs" or "distribution"
        probs = ans.get("probs") or ans.get("distribution") or {}
        if choice == "none":
            return set()
        if choice == "all":
            return set(ALL)
        if choice in TYPES:
            # if we have a full distribution, keep any type close to the top
            if probs:
                top = max((probs.get(t, 0.0) for t in TYPES), default=0.0)
                picked = {t for t in TYPES if probs.get(t, 0.0) >= max(threshold, 0.6 * top)}
                return picked or {choice}
            return {choice}
        return set()

    return predict


def make_laya_noul(agent, threshold=0.5):
    def predict(text):
        res = agent.predict(text, LAYA_NOUL)
        out = set()
        for t in TYPES:
            p = res["answers"][t].get("noul")
            if p is None:
                p = res["answers"][t].get("probability")
            if p is not None and p >= threshold:
                out.add(t)
        return out

    return predict


# --------------------------------------------------------------------------
# GLiNER -- extracts entities ALREADY in the text (wrong task, on purpose)
# --------------------------------------------------------------------------
def make_gliner():
    from gliner import GLiNER

    model = GLiNER.from_pretrained("urchade/gliner_small-v2.1", map_location="cuda")
    labels = [
        "salesperson name",
        "city or location",
        "product name",
        "client or company name",
    ]
    mapping = {
        "salesperson name": P,
        "city or location": L,
        "product name": R,
        "client or company name": C,
    }

    def predict(text):
        ents = model.predict_entities(text, labels, threshold=0.4)
        found = {mapping[e["label"]] for e in ents if e["label"] in mapping}
        # Heuristic A (naive NER): mentioned types are the prediction.
        # This is the opposite of next-slot routing when a type is already filled.
        return found

    return predict


def make_gliner_exclude():
    """ALL minus mentioned types -- slightly closer to 'what's left' but still not a slot model."""
    inner = make_gliner()

    def predict(text):
        mentioned = inner(text)
        if not mentioned:
            return set(ALL)
        remaining = set(ALL) - mentioned
        return remaining or mentioned

    return predict


# --------------------------------------------------------------------------
# Production yes/no verbalizer on a different causal LM
# --------------------------------------------------------------------------
def make_qwen_yesno(model_name):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "classifier_service"))
    from router import LMYesNo  # noqa: E402

    lm = LMYesNo(model_name)
    th = float(os.environ.get("TH", "0.6"))

    def predict(text):
        scores = lm.probs(text)
        return {t for t in TYPES if scores[t] >= th}

    return predict


def main():
    backend = sys.argv[1]
    extra = sys.argv[2] if len(sys.argv) > 2 else None
    quiet = bool(os.environ.get("QUIET"))

    if backend == "laya_choice":
        agent = load_laya(extra)
        predict = make_laya_choice(agent)
        name = f"laya_choice {extra or 'english'}"
    elif backend == "laya_noul":
        agent = load_laya(extra)
        th = float(os.environ.get("TH", "0.5"))
        predict = make_laya_noul(agent, threshold=th)
        name = f"laya_noul {extra or 'english'} th={th}"
    elif backend == "gliner":
        predict = make_gliner()
        name = "gliner_small-v2.1 mentioned-entities"
    elif backend == "gliner_exclude":
        predict = make_gliner_exclude()
        name = "gliner_small-v2.1 ALL-minus-mentioned"
    elif backend == "qwen_yesno":
        if not extra:
            raise SystemExit("qwen_yesno needs a model name")
        predict = make_qwen_yesno(extra)
        name = f"qwen_yesno {extra}"
    else:
        raise SystemExit(f"unknown backend {backend}")

    print(f"=== {name} ===")
    score(CASES, predict, f"{name} / main", quiet=quiet)
    score(HOLDOUT, predict, f"{name} / holdout", quiet=quiet)


if __name__ == "__main__":
    main()
