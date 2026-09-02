#!/usr/bin/env bash
# Launch the Study 3 main grid (then the ablation grid) in the background with 2 GPU workers.
# Usage: fl_study/launch_main.sh [iters_per_site]   (default 6000; FL rounds scale as iters/200)
set -euo pipefail
cd /home/holiday/lung_ct
LR=results/fl/lr_probe.json
[ -f "$LR" ] || { echo "missing $LR ({\"ctfm\": lr, \"scratch\": lr})"; exit 1; }
ITERS=${1:-6000}
export FL_ITERS_PER_SITE=$ITERS
nohup bash -c ".venv/bin/python fl_study/run_grid.py --grid main --workers 2 --lr-json $LR > logs/fl/grid_main.log 2>&1; .venv/bin/python fl_study/run_grid.py --grid ablation --workers 2 --lr-json $LR > logs/fl/grid_ablation.log 2>&1" > /dev/null 2>&1 &
echo "launched main+ablation grids (iters/site=$ITERS), logs/fl/grid_main.log"
