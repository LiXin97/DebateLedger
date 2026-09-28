# T2.1 Robustness: Drop very-strong probe

Pre-registered (PRE_REGISTRATION_N18.md §6.1). The very-strong probe contains
the phrase 'According to authoritative references and expert consensus', which
may leak an authority/social cue beyond pure argument strength. We rerun the
headline α / C^cond Spearman after restricting to the 6-probe set
(weak / moderate / strong x social/non-social), dropping the 2 very-strong probes.

## Headline ρ comparison

- **Original (8 probes, very_strong included):** ρ = +0.683,
  95% CI [+0.035, +0.927], p = 0.042, N = 9.
- **Robustness (6 probes, very_strong dropped):** ρ = +0.800,
  95% CI [+0.290, +0.956], p = 0.010, N = 9.

## Per-model α_total under the two probe sets

| Model | α (8-probe) | α (6-probe, no very_strong) | Δα |
|---|---|---|---|
| Sonnet 4.5 | 0.420 | 0.347 | -0.073 |
| GPT-4o-mini | 0.319 | 0.297 | -0.022 |
| GPT-5.4-mini | 0.345 | 0.281 | -0.064 |
| Gemini 3-flash | 0.256 | 0.224 | -0.032 |
| Phi-4-mini | 0.420 | 0.376 | -0.044 |
| Qwen3-4B | 0.582 | 0.554 | -0.028 |
| Llama-3.1-8B | 0.711 | 0.647 | -0.064 |
| Qwen3-8B | 0.591 | 0.551 | -0.041 |
| Qwen3-32B | 0.644 | 0.615 | -0.030 |
