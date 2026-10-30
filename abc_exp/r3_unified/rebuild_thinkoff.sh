#!/usr/bin/env bash
# Rebuild the thinking-off comparison of the four Qwen3.5/3.6 rows (paper, thinking-off rerun table). Zero API, zero GPU.
# Writes abc_exp/results/thinking_off_rerun/thinkoff_compare_rerun.{json,md}. Run from a working copy with the open tier
# downloaded into it, after rebuild_r3.sh. On the open tier the truncation sensitivities are left out: they need the model
# text. With the gated traces in abc_exp/results/thinking_off_rerun/, set FULL_TEXT=1 to add them.
# usage: bash abc_exp/r3_unified/rebuild_thinkoff.sh     (PYTHON selects the interpreter; default python3)
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3}; E=abc_exp/experiments; RES=$PWD/abc_exp/results; T=$RES/thinking_off_rerun
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
STEMS="qwen3.5-4b qwen3.5-9b qwen3.6-27b-fp8 qwen3.6-35b-a3b-fp8"
mkdir -p "$W/old" "$W/new" "$W/filt"
for d in new filt; do for f in "$RES"/debate_traces_*.jsonl "$RES"/sa_causal_*.jsonl; do ln -s "$f" "$W/$d/"; done; done
for s in $STEMS; do for k in debate_traces sa_causal; do
  ln -s "$RES/${k}_vllm_${s}.jsonl" "$W/old/"
  for d in new filt; do ln -sfn "$T/${k}_vllm_${s}_thinkoff.jsonl" "$W/$d/${k}_vllm_${s}.jsonl"; done
done; done
"$PY" $E/replay_1255_reproduction.py --results-dir "$W/new" --tag rerun > "$W/replay.log"
DT=()
if [ "${FULL_TEXT:-0}" = 1 ]; then
  "$PY" $E/thinkoff_truncation.py --dir "$T" --suffix _thinkoff --tag rerun --out-dir "$T" --write-filtered "$W/filtered" > "$W/trunc.log"
  OV=(); for s in $STEMS; do OV+=(--override "vllm_$s=$W/filtered/debate_traces_vllm_${s}_droptrunc.jsonl"); done
  "$PY" $E/replay_1255_reproduction.py --results-dir "$W/filt" --tag droptrunc "${OV[@]}" > "$W/filt.log"
  DT=(--droptrunc-replay "$W/filt/replay_1255_reproduction_droptrunc.json")
fi
"$PY" $E/thinkoff_compare.py --old-dir "$W/old" --new-dir "$W/new" --old-replay "$RES/replay_1255_reproduction_r3unified.json" \
  --new-replay "$W/new/replay_1255_reproduction_rerun.json" ${DT[@]+"${DT[@]}"} --tag rerun --out-dir "$T" > /dev/null
echo "done: $T/thinkoff_compare_rerun.md"
