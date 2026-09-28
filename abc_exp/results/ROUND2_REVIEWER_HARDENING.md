# Round-2 Reviewer Hardening

Zero-cost analyses computed from checked-in artifacts only. These are rebuttal/camera-ready diagnostics, not new confirmatory tests.

## Held-Out-Family LoFO Predictive Check

Post-hoc leave-one-family-out linear predictive check on family means; intervals are unbounded Gaussian prediction intervals fit on the other six families.

Predicted-vs-observed Spearman: +0.821 (n=7, p=0.0234). 80% PI coverage: 5/7; 95% PI coverage: 6/7. Mean absolute rank error: 0.857.

| Held-out family | alpha | Observed C^cond | Predicted C^cond | 80% PI | 95% PI | Rank err |
|---|---:|---:|---:|---:|---:|---:|
| Anthropic | 0.4504 | 2.15% | 6.77% | [-1.96, +15.50] | [-9.03, +22.58] | 1 |
| DeepSeek | 0.1129 | 0.00% | -4.13% | [-16.01, +7.75] | [-25.64, +17.38] | 0 |
| Google | 0.2875 | 0.29% | 2.70% | [-7.03, +12.43] | [-14.92, +20.31] | 0 |
| Meta | 0.7737 | 8.46% | 18.94% | [+9.97, +27.92] | [+2.69, +35.20] | 2 |
| OpenAI | 0.3461 | 1.52% | 4.02% | [-5.39, +13.43] | [-13.02, +21.07] | 0 |
| Phi | 0.4686 | 10.24% | 5.93% | [-2.90, +14.76] | [-10.06, +21.92] | 2 |
| Qwen | 0.6987 | 19.75% | 8.21% | [+2.13, +14.29] | [-2.81, +19.23] | 1 |

## Comparator / Proxy Bakeoff

Numerical comparator/proxy bakeoff from local artifacts. Pandey-circuit and exact Engels-gap scores are not present, so proxies are labeled explicitly.

| Score | Role | Model-row Spearman vs C^cond | Family Spearman vs C^cond | Note |
|---|---|---:|---:|---|
| `alpha_tot` | selection-time fingerprint | +0.829 (n=14, p=0.000243) | +0.893 (n=7, exact p2=0.0123, p1-pos=0.00615) | headline selection-time measurement; measured before downstream debate sweep |
| `engels_capability_pressure_proxy` | Engels-style capability proxy (not exact Engels gap) | +0.803 (n=14, p=0.000543) | +0.821 (n=7, exact p2=0.0341, p1-pos=0.0171) | defined as 1 - initial-majority accuracy; exact heterogeneous capability-gap/overseer score is not in local artifacts |
| `debate_revision_proxy` | raw revision-quantity baseline | +0.456 (n=14, p=0.101) | +0.714 (n=7, exact p2=0.0881, p1-pos=0.044) | debate flip-rate / majority-flip proxy; measured from debate traces or older conditional-collapse table |
| `social_proxy_alpha_social` | Sharma/Perez-style social-pressure proxy | +0.873 (n=14, p=4.48e-05) | +0.893 (n=7, exact p2=0.0123, p1-pos=0.00615) | mean flip rate on social-attributed counterarguments, using sa_causal default rows where available and alpha_split fallback otherwise |
| `social_proxy_lift` | social-over-solo lift proxy | +0.026 (n=14, p=0.929) | -0.071 (n=7, exact p2=0.906, p1-neg=0.453) | difference between social and solo probe flip rates; narrower sycophancy/conformity proxy than alpha_tot |
| `initial_disagreement_rate` | Tang-style runtime disagreement proxy | +0.928 (n=6, p=0.00767) | +0.866 (n=3, exact p2=0.667, p1-pos=0.333) | available only for six saved post-A6 raw-trace rows; measured after sampling three initial answers |
| `r1_majority_changed_rate` | Tang-style R1 trajectory proxy | +0.986 (n=6, p=0.000309) | +0.866 (n=3, exact p2=0.667, p1-pos=0.333) | available only for six saved post-A6 raw-trace rows; runtime diagnostic, not selection-time measurement |

Partial alpha after capability-pressure proxy: +0.658 (n=14, p=0.0144).
Partial alpha after capability-pressure + raw-revision proxy: +0.667 (n=14, p=0.0178).

Unavailable exact scores:
- Pandey-2026 mechanistic sycophancy/circuit activations: no local *pandey* artifact or activation-cache score found.
- Exact Engels-2025 capability-gap forecast: local traces are homogeneous three-agent debates, not overseer/worker capability-gap panels.

## Alpha-Lite Retrospective Cost Check

Zero-API retrospective alpha-lite check. N=14 uses default-condition all-agent alpha because older sealed rows lack per-agent raw rows; strict one-agent alpha-lite is available only on six post-A6 raw lanes.

| Check | Spearman vs C^cond | Scope |
|---|---:|---|
| N=14 default-condition reduction | +0.893 (n=7, exact p2=0.0123, p1-pos=0.00615) | all headline rows; sealed rows are aggregate, not per-agent |
| Post-A6 strict one-agent alpha-lite | +0.866 (n=3, exact p2=0.667, p1-pos=0.333) | 6 raw rows across three families |

The available N=14 default-condition reduction preserves the headline family rank correlation, but a strict N=14 one-agent alpha-lite replay cannot be claimed without recovering per-agent raw rows for the sealed lanes.

## Qwen3-32B Sensitivity

Qwen3-32B corrected-pool holdout: alpha=0.6767, C^cond=10.96%, correction=30.71% over 200 debates.

| Sensitivity | Spearman | Note |
|---|---:|---|
| Model row N=15 | +0.822 (n=15, p=0.000169) | Qwen3-32B appended as a model row |
| Family G=7, Qwen updated | +0.893 (n=7, exact p2=0.0123, p1-pos=0.00615) | Qwen3-32B averaged into Qwen family |
| Stress-family G=8 | +0.857 (n=8, exact p2=0.0107, p1-pos=0.00536) | Qwen3-32B shown as separate post-hoc stress row |

## Same-Pool LOMO Gate Audit

The exact 6,525-row pilot-gated LOMO records are not present in this checkout; only the aggregate Table 17 numbers and a separate 1,255-record saved-trace replay cohort are available. Therefore a strict matched-tau LOMO DG/DRS comparison cannot be honestly claimed from local files without recovering the missing row-level pool.

Available saved-trace replay cohort: 2438 records across 13 sources.
DisagreementGate on that available cohort: delta=-3.28pp, prevented=80, lost=160, net=-80, gate=33.8%.
