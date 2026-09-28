# T2.4 Robustness: Log-ratio S/A normalization

Pre-registered (PRE_REGISTRATION_N18.md §6.4). The headline S/A normalization
uses `s_soc / (|s_soc| + |s_arg| + eps)`; we report a log-ratio variant
`log((s_soc + eps) / (s_arg + eps))` for completeness.

| Model | S | A | S/A (orig) | S/A (log) |
|---|---|---|---|---|
| Sonnet 4.5 | +0.047 | +0.327 | +0.125 | -1.928 |
| GPT-4o-mini | +0.010 | +0.134 | +0.069 | -2.509 |
| GPT-5.4-mini | +0.008 | +0.272 | +0.030 | -3.375 |
| Gemini 3-flash | -0.014 | +0.152 | -0.085 | -5.028 |
| Phi-4-mini | +0.063 | +0.282 | +0.183 | -1.483 |
| Qwen3-4B | +0.030 | +0.168 | +0.149 | -1.711 |
| Llama-3.1-8B | +0.030 | +0.472 | +0.060 | -2.713 |
| Qwen3-8B | +0.010 | +0.164 | +0.057 | -2.709 |
| Qwen3-32B | +0.055 | +0.170 | +0.242 | -1.126 |

Spearman rank consistency across the 9 models: ρ = +0.983 (p = 0.000).
