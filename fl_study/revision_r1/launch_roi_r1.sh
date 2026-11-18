#!/usr/bin/env bash
# Inference-only revision analyses (one GPU process, resumable). Sentinel: logs/fl_r1/R1_ROI_DONE
set -euo pipefail
cd /home/holiday/lung_ct
R=fl_study/revision_r1
mkdir -p logs/fl_r1 results/fl_r1/roi
rm -f logs/fl_r1/R1_ROI_DONE
nohup bash -c "
  .venv/bin/python $R/eval_roi_r1.py --perturb --tags-file $R/tags_roi_perturb.txt > logs/fl_r1/roi_perturb.log 2>&1
  .venv/bin/python $R/eval_roi_r1.py --tags-file $R/tags_roi_baseline.txt > logs/fl_r1/roi_baseline.log 2>&1
  date > logs/fl_r1/R1_ROI_DONE
" > /dev/null 2>&1 < /dev/null &
echo "launched ROI analyses; logs/fl_r1/roi_perturb.log"
