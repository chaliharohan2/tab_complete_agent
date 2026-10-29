# Classifier service: approaches and serving design

This is the GPU-hosted next-entity router (`service.py` + `router.py` + `rules.py`).
Given the text typed so far, it returns which of `{person, location, product, client}`
could come **next** (multi-label; possibly none).

Nothing here is trained. Search-term extraction (the partial word, e.g. `"del"`)
is **not** this service's job — that is a string split in the backend
(`backend/text_utils.py` / `rules.split_context`).

Eval harness: `eval/eval_service.py` (hits the deployed `/classify`),
`eval/test_cases.py` (main set), `eval/test_cases_holdout.py` (not used while
tuning). Rule-only checks: `eval/test_rules.py`.

---

## Current design (what is deployed)

Two layers, first match wins:

1. **Rules** (`rules.py`) — deterministic, ~0ms. Covers the obvious syntax so
   the LLM is not asked when it does not need to be.
2. **Qwen3-8B yes/no scorer** (`router.py` / `LMYesNo`) — few-shot, one batched
   forward pass, independent `P(yes)` per type. Default threshold `0.6`.

Default model: `Qwen/Qwen3-8B` (`DEFAULT_MODEL`). Override with `ROUTER_MODEL`.
Threshold: `ROUTER_THRESHOLD` (default `0.6`).

Last measured (after the rule expansion): main **65/65**, holdout **31/32**,
LLM-path latency ~130–200ms, overall median ~50ms because most obvious cases
are rules.

---

## Approaches we tried (and why they were dropped)

All zero-shot / few-shot. No fine-tuning.

| Approach | Model | Device | Latency | Result |
|---|---|---|---|---|
| Sentence-embedding cosine sim | `all-MiniLM-L6-v2` (22M) | CPU | ~40ms | Poor. Collapses toward one label regardless of syntax. |
| Sentence-embedding cosine sim | `BAAI/bge-small-en-v1.5` | CPU | ~100ms | Poor. Same collapse, even with an instruction prefix. |
| Zero-shot NLI | `typeform/distilbert-base-uncased-mnli` | CPU | 150–500ms | Unstable. e.g. "Where was this sold" → product 0.99 over location 0.42. |
| Zero-shot NLI | `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` (~70M) | CPU | 1–2s | Accurate enough, unusable latency. DeBERTa-v3 disentangled attention is expensive on CPU. |
| Zero-shot NLI | same xsmall | GPU | **5–8ms** | Fast, but framed as "what is this sentence *about*", not "what value comes *next*". Over-associated generic `"Show me the sales for"` with `client`. Failed obvious next-slot cases (`units of` → location, `sold the most in` → product, `In delhi ... sales of` → location+client). |
| Instruct generation (JSON) | `Qwen2.5-0.5B-Instruct` | GPU | 95–130ms | JSON was parseable; accuracy worse than NLI. Missed "who sold" → person, "where sold" → location. Unreliable span extraction. |
| Instruct generation | Qwen2.5 / Qwen3 &lt;2B, next-token label / free-text "Next: person" | GPU | varies | Too small to reliably do the "who/where/of/in" inferences few-shot. |
| Causal LM value likelihood | score how probable example *values* of each type are as the literal continuation | GPU | slower | Explored in `eval/run_eval.py` (`lm_values`). Did not beat yes/no scoring. |
| **Yes/no verbalizer (kept)** | **`Qwen/Qwen3-8B`** | **GPU** | **~130–200ms** | Independent multi-label `P(yes)` per type. Good on paraphrases the NLI model missed. Combined with rules for the obvious syntax. |

Takeaways that are still true:

- Bi-encoder similarity is the wrong tool for *slot* routing. It measures
  topical overlap, not "what kind of value comes next."
- NLI is a decent topic classifier and is very fast on GPU, but "this example
  is about a client" is the wrong question for `"Show me the sales for"`.
- Sub-2B instruct models do not reliably do this few-shot. 8B does, if you
  stop asking it to generate.
- Generation (JSON / free-text labels) is slower *and* less calibrated than
  reading two logits.

The earlier NLI-era writeup is in `docs/approach.md` (now historical).

---

## Why Qwen3-8B is ~130ms, not ~1s+

We did **not** replace or retrain the last layer. We stopped using the model
as a generator and used the existing `lm_head` as a 2-class classifier
(a verbalizer). That is the "classification layer on top."

A normal 8B instruct call is expensive because of:

1. **Prefill** — the whole prompt (header + few-shot + user text) is encoded
   every request.
2. **Decode** — tokens are produced one-by-one; each step is a full forward
   pass.
3. **Four labels** — generating `"person, product"` (or running four
   `generate()` calls) stacks that cost.

An 8B decode loop for even 10–20 tokens is hundreds of milliseconds. The
scorer removes the loop.

### 1. Next-token logits, not generated text

Few-shot lines look like:

```
Text: "Get me the revenue numbers for"
Could the next words be a location? yes
```

At request time we feed only up to the question and **do not call
`generate()`**. `LMYesNo.probs` reads the last-token logits and keeps two
vocabulary rows — the token ids of `" yes"` and `" no"` (leading space
matters for Qwen BPE):

```python
yn = logits[i, lens[i] - 1, [self.yes, self.no]].float()
p_yes = torch.softmax(yn, -1)[0].item()
```

`logits[i, last, :]` is `lm_head` applied to the last hidden state — a vector
over the full vocab (~150k). Softmax over those two entries is `P(yes | text,
question)`. No new weights. One forward pass over ~20–40 suffix tokens. Zero
decode steps.

This is the standard LM-as-classifier / PET verbalizer trick.

### 2. Few-shot prefix is KV-cached once

The few-shot block is long (header + many examples × 4 yes/no lines).
Re-encoding it on every keystroke would dominate latency.

At startup (`LMYesNo.__init__`) we run the prefix once and keep
`past_key_values` (K/V for every layer, every prefix token). Each request
only forwards the short suffix:

```
Text: "<user text>"
Could the next words be a location?
```

Attention over the prefix is reused. `position_ids` for the suffix start at
`prefix_len` so RoPE stays aligned with the cache. The cache is
`deepcopy`'d per request because the forward pass appends to it.

### 3. All four types in one batched forward

The four questions share that prefix. We `batch_repeat_interleave(4)` the
cached K/V and run **one** batched forward instead of four sequential ones.
Four independent `P(yes)` scores come out of one kernel — that is why the
output is multi-label (person and product can both be high).

### 4. Rules skip the model

Obvious syntax never hits the GPU. See the next section. That is why eval
median latency is ~50ms even though an LLM hit is still ~130–200ms.

### Implementation pitfalls (do not "simplify" these)

- Tokenize the suffix with `add_special_tokens=False`. A second BOS would
  break continuity with the cached prefix.
- `position_ids` **must** continue from `prefix_len`. Default 0-based
  positions with a prefix cache produce garbage RoPE and garbage scores.
- Deepcopy the cache before each request. In-place append will corrupt the
  startup cache.
- Yes/no token ids are `" yes"` / `" no"` (leading space), not `"yes"` /
  `"no"`.
- Pad on the right and read `logits[i, lens[i] - 1, ...]` (last *real*
  token), not the padded end.

---

## Rule layer (`rules.py`)

First match wins. `rule_types(split_context(text))` returns a list of types
or `None` (fall through to the LLM).

1. **Explicit type at the slot** — `"for the client"`, `"in the city of"`,
   `"for the sales person"`. Keyword list in `TYPE_KEYWORDS` (longest first).
2. **Constructions**
   - trailing `in` / `at` → location
   - `units|stock|inventory|orders of` → product
   - `sold|closed|handled|earned by` → person
   - `sales of <X> by` → person (distinct from generic `"split by"`)
   - `sold|billed|invoiced to` → client
   - `shipped|sent|delivered to` → client + location
   - `sales of <X> to` → client
3. **Already-used type + new slot** — leading `"In Delhi, ..."` marks
   location used; a type keyword used as the subject (`"which product ...
   for"`) marks that type used. Trailing `for|of|from|by` then returns the
   *remaining* types. This is what fixes `"In delhi what were the sales of"`
   (not location) and `"Top selling products for"` (not product).
4. **Bare `for` / `of`** with no other signal → all four types. This is what
   removes the old client over-association on `"Show me the sales for"`.

`split_context` drops a trailing partially-typed value (`"...for del"` →
`"...for"`) so rules see the committed slot, not `"del"`.

When changing rules: run `eval/test_rules.py` locally (no GPU), then
`eval/eval_service.py` against the deployed service. Do not copy holdout
sentences into `FEW_SHOT` or into new rules — that is what the holdout is
for.

---

## What to touch when changing behavior

| Want to change | Where |
|---|---|
| Obvious syntax / prepositions / type keywords | `rules.py` |
| Ambiguous paraphrases the rules should not own | `FEW_SHOT` / `QUESTIONS` / `HEADER` in `router.py` |
| How aggressive "include this type" is | `DEFAULT_THRESHOLD` (0.6) |
| Model | `DEFAULT_MODEL` / `ROUTER_MODEL` — keep the yes/no scorer; do not go back to `generate()` without a latency budget |
| Deploy | `python ops/deploy_classifier.py` (uploads these three files, restarts uvicorn on the GPU box, port 8800) |

After a `FEW_SHOT` or model change the prefix KV cache is rebuilt on
startup (8B load is 1–2 minutes). After a rules-only change you still have
to restart so the GPU process imports the new file.
