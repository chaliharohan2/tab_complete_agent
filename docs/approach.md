# Model selection writeup

## Problem framing

The task splits into two independent sub-problems:

1. **Entity-type routing**: given the text typed so far, which of
   `{person, location, product, client, none}` are plausible right now?
   This is a small, closed-set classification problem.
2. **Search-term extraction**: what substring should be sent to the fuzzy
   search (e.g. `"del"`)? In practice this is just "the partial word
   currently being typed" -- a string-parsing problem, not an ML problem.

Framed this way, a generative LLM solves a harder, more open-ended problem
(free text generation + JSON formatting + span extraction) than the task
actually needs. The recommendation below leans on a purpose-built zero-shot
*classifier* instead, with a plain string-split rule for #2.

## Models tried

All models used zero-shot / few-shot only -- nothing was trained or
fine-tuned.

| Approach | Model | Params | Device | Latency | Accuracy |
|---|---|---|---|---|---|
| Sentence-embedding similarity | `all-MiniLM-L6-v2` | 22M | CPU | ~40ms | Poor -- collapses toward one label (e.g. "person") regardless of content |
| Sentence-embedding similarity | `BAAI/bge-small-en-v1.5` (w/ instruction prefix) | 33M | CPU | ~100ms | Poor -- same collapse problem |
| Zero-shot NLI cross-encoder | `typeform/distilbert-base-uncased-mnli` | 66M | CPU | ~150-500ms | Poor/unstable -- e.g. scored `product=0.99` over `location=0.42` for "Where was this sold" |
| Zero-shot NLI cross-encoder | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | 70M | CPU | ~1000-2000ms | **Good** |
| Zero-shot NLI cross-encoder | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | 70M | **GPU** | **~5-8ms** | **Good** (same as CPU, just fast) |
| Few-shot instruct LLM | `Qwen2.5-0.5B-Instruct` | 0.5B | GPU | ~95-130ms | Worse than NLI -- missed "Who sold this item" -> person, "Where was this sold" -> location; also unreliable span extraction |

Takeaways:

- **Bi-encoder embedding similarity isn't discriminative enough** for this
  task zero-shot, with two different models tested. It's fast but not
  usable as-is.
- **The one CPU-fast NLI model tested (DistilBERT-based) was inaccurate.**
- **The one accurate NLI model (DeBERTa-v3-xsmall) is too slow on CPU**
  (~1-2s/call) -- not because the model is big, but because DeBERTa-v3's
  disentangled attention is expensive per layer on CPU. On GPU it drops to
  single-digit milliseconds.
- **A small instruct LLM (Qwen2.5-0.5B) underperformed the NLI classifier**
  on both speed and accuracy, even on GPU. It's solving a harder problem
  (generation + JSON formatting) than the task requires, and a 0.5B model
  doesn't reliably make the "who sold X -> person" / "where was X sold ->
  location" kind of inference zero-shot.

## Decision

**`MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33`, hosted on GPU**, for
entity-type routing. Plain string-splitting (no model) for search-term
extraction. See `backend/config.py` for the exact label descriptions and
routing thresholds, and `README.md` for how to tune them.
