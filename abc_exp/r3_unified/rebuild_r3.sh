#!/usr/bin/env bash
# Rebuild the paper's numbers from the released traces at Round 3 with alphabetic ties. Zero API, zero GPU.
# It rewrites trace files in place: run it on a working copy of this repository with the open tier downloaded into it.
# usage: bash abc_exp/r3_unified/rebuild_r3.sh --in-place     (PYTHON selects the interpreter; default python3)
set -euo pipefail
[ "${1:-}" = "--in-place" ] || { echo "This rewrites traces in place. Use a working copy, then pass --in-place."; exit 2; }
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python3}; U=abc_exp/r3_unified; E=abc_exp/experiments; S=abc_exp/scripts
run() { echo "== $*"; "$PY" "$@"; }
run $U/bulk_alpha_rows.py .
run $U/recode_r3.py .
run $U/holdout_32b_summary.py .
run $S/recompute_headline_n18.py --traces-dir abc_exp/results --alpha-dir abc_exp/results --out abc_exp/results/cohort_n14_headline.json --ccond-endpoint r3_alpha
run $U/d52_update_rows.py .
run $E/family_level_headline_aggregation.py
run $E/taxonomy_sweep.py
run $E/reviewer_hardening_analyses.py
run $E/n14_reviewer_hardening_artifacts.py
run $E/family_partial_spearman.py
run $E/leave_one_qwen_row_sensitivity.py
# The strict-parser audit needs the model text, so it runs only on the gated full-text tier (FULL_TEXT=1).
# On the open tier the released parser_cell_stability_audit.* are kept.
if [ "${FULL_TEXT:-0}" = 1 ]; then run $E/parser_cell_stability_audit.py; fi
run $E/replay_1255_reproduction.py --results-dir abc_exp/results --tag r3unified
run $E/round2_reviewer_hardening.py
run $E/round4_camera_ready_hardening.py
run $S/freeze_extension_R3.py
# the power script takes its planning values from the two partial correlations above
PR=$("$PY" -c "import json;print(round(json.load(open('abc_exp/results/family_partial_spearman.json'))['family_partial_spearman']['rho'],3))")
ER=$("$PY" -c "import json;print(round(json.load(open('abc_exp/results/rebuttal_extension_R3_frozen.json'))['family_partial_after_init_acc']['rho'],3))")
sed -i "s/^    planning_rho = .*/    planning_rho = $PR/; s/^    extension_rho = .*/    extension_rho = $ER/" $S/rebuttal_q1_pairwise_and_power.py
run $S/rebuttal_q1_pairwise_and_power.py
echo "done"
