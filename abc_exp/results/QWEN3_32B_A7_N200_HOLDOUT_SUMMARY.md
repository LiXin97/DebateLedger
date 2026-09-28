# Qwen3-32B (local vLLM) Holdout Summary

This is a post-hoc holdout/expansion artifact and is not folded into the sealed headline table unless explicitly relabeled.

## Alpha

- Rows: 1800 over 200 questions.
- Grid QC: 0 missing rows and 0 duplicate triples on the observed condition-agent grid.
- Alpha total: 0.6767.
- Alpha by condition: anti_argument=0.7002, anti_social=0.6754, default=0.6546.
- Cost: $0.0000.

## Debate

- Debates: 200.
- Debate QC: 0 duplicate question rows.
- Initial/final majority accuracy: 36.50% -> 52.00%.
- Majority collapses: 8 / 73 at-risk (10.96%).
- Majority corrections: 39 / 127 initially-wrong (30.71%).
- Majority flip rate: 37.00%.
- Agent-level collapse/correction events: 17 / 72.
- Cost: $0.0000.

## Provider Provenance

- Alpha providers: `{}`.
- Debate providers: `{}`.

## Pair QC

- Alpha/debate question overlap: 200 shared questions; 0 alpha-only; 0 debate-only.
