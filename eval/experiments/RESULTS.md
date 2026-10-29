# Alternative-model bake-off (2026-10-29)

Production router (`classifier_service/`, Qwen3-8B + rules) was **not** changed.
Sweep harness: `eval/experiments/sweep.py`. Raw rows: `eval/experiments/sweep_results.jsonl`.
Same verbalizer / NLI next-slot template against `eval/test_cases.py` (65) and
`eval/test_cases_holdout.py` (32). Metric: required ⊆ pred ⊆ required ∪ optional.

**Verdict: keep Qwen3-8B yes/no + rules.** No smaller or same-size model matched
it on model-only accuracy. The closest smaller model is Qwen3-4B-Instruct-2507
(76.9% / 71.9% vs 8B 87.7% / 87.5%). Rules hide some of that gap but not all
(4B+rules 95.4% / 90.6% vs 8B+rules 100% / 96.9%).

## Research: what we looked for

Sources checked before expanding the sweep (Oct 2026):

- [TinyWeights SLM table](https://tinyweights.dev/models/)
- [On-device SLMs 2026](https://aicraftguide.com/article/on-device-slm-phi4-gemma3-qwen3-smollm3-2026)
- [vdf.ai best SLMs](https://vdf.ai/blog/best-small-language-models/)
- [Awesome Agents under-10B leaderboard](https://awesomeagents.ai/leaderboards/small-language-model-leaderboard/)
- [KDnuggets HF SLM list](https://www.kdnuggets.com/best-small-language-models-on-hugging-face-right-now)
- Hugging Face zero-shot classification recs (BART-MNLI, ModernBERT-large-zeroshot)
- Model cards: Qwen3.5-4B, Qwen3-4B-Instruct-2507, Phi-4-mini, Gemma 3 4B,
  SmolLM3-3B, Granite 4.1 3B, MiniCPM5-2B, LFM2.5, Olmo 3 7B, Ministral 3 3B

Reviews that claim a 3–4B can “rival 7–9B” are about MMLU / GSM8K / IFEval, not
next-slot routing. We still ran every open-weight candidate that (a) is ≤8B,
(b) loads in current `transformers` without a Hub gate, and (c) can score
yes/no or NLI.

## Full scoreboard (model-only @ 0.6 unless noted)

Production + rules (unchanged, live `:8800`): **65/65 = 100%** main, **31/32 = 96.9%** holdout, blended median ~50ms.

| Model | Size | Main | Holdout | +rules main/hold | LLM ms | Keep? |
|---|---|---|---|---|---|---|
| **Qwen3-8B yes/no (current)** | 8B | **57/65 = 87.7%** | **28/32 = 87.5%** | **100% / 96.9%** | 136 | **yes** |
| Qwen3-4B-Instruct-2507 | 4B | 50/65 = 76.9% | 23/32 = 71.9% | 95.4% / 90.6% | 99 | no — closest smaller, still −11 / −16 pts |
| Granite 4.1 3B | 3B | 50/65 = 76.9% | 19/32 = 59.4% | 96.9% / 87.5% | 73 | no — ties 4B on main, collapses on holdout |
| Qwen2.5-7B-Instruct | 7B | 49/65 = 75.4% | 24/32 = 75.0% | 93.8% / 93.8% | 107 | no — similar size, worse than 8B |
| Qwen3.5-4B (seq.) | 4B | 41/65 = 63.1% | 26/32 = 81.2% | 92.3% / 93.8% | 239 | no — hybrid attn, slower, weaker main |
| SmolLM3-3B | 3B | 41/65 = 63.1% | 19/32 = 59.4% | 95.4% / 84.4% | 62 | no |
| MiniCPM5-2B | 2B | 35/65 = 53.8% | 20/32 = 62.5% | 90.8% / 87.5% | 55 | no |
| Qwen2.5-3B-Instruct | 3B | 31/65 = 47.7% | 13/32 = 40.6% | 93.8% / 84.4% | 62 | no |
| LFM2.5-2.6B | 2.6B | 31/65 = 47.7% | 15/32 = 46.9% | 95.4% / 87.5% | 124 | no |
| Phi-4-mini-instruct | 3.8B | 29/65 = 44.6% | 11/32 = 34.4% | 90.8% / 78.1% | 80 | no |
| Qwen3-1.7B-Base | 1.7B | 29/65 = 44.6% | 11/32 = 34.4% | 90.8% / 75.0% | 51 | no |
| Qwen3.5-2B (seq.) | 2B | 26/65 = 40.0% | 19/32 = 59.4% | 90.8% / 84.4% | 127 | no |
| Qwen2.5-1.5B-Instruct | 1.5B | 26/65 = 40.0% | 12/32 = 37.5% | 92.3% / 81.2% | 35 | no |
| DeBERTa-v3-base zeroshot NLI | 184M | 16/65 = 24.6% | 5/32 = 15.6% | 90.8% / 78.1% | 7 | no — rules carry it |
| Qwen3.5-0.8B (seq.) | 0.8B | 15/65 = 23.1% | 9/32 = 28.1% | 89.2% / 81.2% | 87 | no |
| SmolLM2-1.7B-Instruct | 1.7B | 10/65 = 15.4% | 6/32 = 18.8% | 83.1% / 75.0% | 48 | no |
| Gemma 3 1B IT | 1B | 9/65 = 13.8% | 5/32 = 15.6% | 84.6% / 71.9% | 22 | no |
| LFM2.5-1.2B-Instruct | 1.2B | 9/65 = 13.8% | 6/32 = 18.8% | 80.0% / 71.9% | 59 | no |
| BART-large-MNLI | 407M | 4/65 = 6.2% | 4/32 = 12.5% | 84.6% / 78.1% | 23 | no |
| Laya choice | 0.4B | 12/65 = 18.5% | 2/32 = 6.2% | — | 17 | no (prior run) |
| Laya noul | 0.4B | 8/65 = 12.3% | 4/32 = 12.5% | — | 19 | no (prior run) |
| spaCy NER mentioned | 12MB | 8/65 = 12.3% | 7/32 = 21.9% | — | 1 | no (prior run) |
| spaCy last-ADP POS | 12MB | 30/65 = 46.2% | 17/32 = 53.1% | — | 1 | already in `rules.py` |
| MiniLM / BGE cosine | 22–33M | poor | — | — | ~40–100 | no (APPROACH.md) |
| DistilBERT-MNLI | 66M | unstable | — | — | 150–500 | no (APPROACH.md) |
| DeBERTa-v3-xsmall NLI | 70M | aboutness fail | — | — | 5–8 GPU | no (APPROACH.md) |
| Qwen2.5-0.5B JSON gen | 0.5B | worse than NLI | — | — | 95–130 | no (APPROACH.md) |

Could not load (not a quality miss):

| Model | Why |
|---|---|
| Gemma 3 4B IT | gated Hub repo, no HF token on the GPU box |
| Llama 3.2 3B Instruct | gated Hub repo |
| Ministral 3 3B Instruct | `transformers` on the box missing the architecture configs |
| ModernBERT-large zeroshot | unauthenticated Hub download hung; same NLI template as DeBERTa-base (26%) |
| DeBERTa-v3-large zeroshot | same NLI template as the 26% base; skipped |
| Clef-flash 9B | larger than 8B, same System-1 API as Laya (18%) |

## What the numbers say

1. **Scale still wins this task.** Model-only accuracy jumps at 4B (Granite / Qwen3-4B-Instruct ~77%) and again at Qwen3-8B (88%). Reviews that put Qwen3.5-4B or Phi-4-mini next to 8B on MMLU do not transfer. Phi-4-mini is 44.6% here — tied with Qwen3-1.7B.

2. **Qwen3.5-4B is not a substitute.** Hybrid linear attention cannot use the batched KV-cache verbalizer, so it is *slower* than 8B (239ms vs 136ms) and 25 points worse on main.

3. **Granite 4.1 3B looks tempting on the main set and then fails holdout** (76.9% → 59.4%). That is memorizing the few-shot style, not routing.

4. **Same-size 7B is not better.** Qwen2.5-7B (75%) and Olmo-3-7B (31%) both lose to Qwen3-8B. Generation / year of the checkpoint matters more than parameter count once you are already at 7–8B.

5. **NLP encoders are the wrong question.** NLI / MNLI / NER / POS / GLiNER / Laya all classify *aboutness* or *already-mentioned spans*. Next-slot is “what value is missing.” DeBERTa-base 24.6%, BART-MNLI 6.2%. Their +rules scores are just `rules.py`.

6. **Sub-2B still cannot few-shot this.** 0.8B–1.7B sit at 14–45%. Rules then do the work and the LLM layer is dead weight.

## Cleanup (GPU box)

Safe to delete — downloaded for this sweep, not used by production:

```
rm -rf ~/.cache/huggingface/hub/models--convaiinnovations--laya
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
rm -rf ~/.cache/huggingface/hub/models--google--gemma-3-1b-it
rm -rf ~/.cache/huggingface/hub/models--microsoft--Phi-4-mini-instruct
rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM3-3B
rm -rf ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B-Instruct
rm -rf ~/.cache/huggingface/hub/models--ibm-granite--granite-4.1-3b
rm -rf ~/.cache/huggingface/hub/models--openbmb--MiniCPM5-2B
rm -rf ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B
rm -rf ~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct
rm -rf ~/.cache/huggingface/hub/models--allenai--Olmo-3-7B-Instruct
rm -rf ~/.cache/huggingface/hub/models--facebook--bart-large-mnli
rm -rf ~/.cache/huggingface/hub/models--MoritzLaurer--ModernBERT-large-zeroshot-v2.0
rm -rf ~/.cache/huggingface/hub/models--MoritzLaurer--deberta-v3-base-zeroshot-v2.0
rm -rf ~/tab_complete_poc_bench/exp_venv
```

Do **not** delete `models--Qwen--Qwen3-8B` (production). Optional keep:
`Qwen3-1.7B-Base`, `Qwen3.5-2B` if you want to re-run the small-model baseline.
