#!/usr/bin/env bash
# Run the whole pipeline in order. Every stage skips work whose outputs already exist (use --force to redo).
#   bash scripts/run_all.sh                                   # full data, configs/default.yaml
#   bash scripts/run_all.sh configs/default.yaml --set tower.epochs=2
#   PYTHON=.venv/bin/python bash scripts/run_all.sh configs/smoke.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${1:-configs/default.yaml}
if [ $# -gt 0 ]; then shift; fi
PY=${PYTHON:-python}

run() {
  local script=$1; shift
  echo ""
  echo "=================== $script $(date +%H:%M:%S)"
  "$PY" "scripts/$script" --config "$CONFIG" "$@"
}

ARGS=("$@")
run 00_prepare.py       ${ARGS[@]+"${ARGS[@]}"}
run 01_eda.py           ${ARGS[@]+"${ARGS[@]}"}
run 02_features.py      ${ARGS[@]+"${ARGS[@]}"}
run 02b_baselines.py    ${ARGS[@]+"${ARGS[@]}"}
run 02c_prior_ncf.py    ${ARGS[@]+"${ARGS[@]}"}
run 02d_window_ablation.py ${ARGS[@]+"${ARGS[@]}"}
run 03_train_tower.py   ${ARGS[@]+"${ARGS[@]}"}
run 04_score_tower.py   ${ARGS[@]+"${ARGS[@]}"}
run 05_train_ranker.py  ${ARGS[@]+"${ARGS[@]}"}
run 06_predict.py       ${ARGS[@]+"${ARGS[@]}"}
run 08_no_tower_ablation.py ${ARGS[@]+"${ARGS[@]}"}
run 07_diagnostics.py   ${ARGS[@]+"${ARGS[@]}"}
echo ""
echo "done. results in outputs/<scope>/results (summary_test.csv), figures in outputs/<scope>/figures"
