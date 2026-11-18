#!/usr/bin/env bash
# Launch the revision-1 training queue in the background (2 GPU workers, resumable: finished tags are skipped).
# Order: E1 extended budget -> E4 FedBN held-out -> E2 fixed-split replications -> E3 FedProx mu grid.
# Sentinel logs/fl_r1/R1_QUEUE_DONE is written when the whole chain has finished.
set -euo pipefail
cd /home/holiday/lung_ct
R=fl_study/revision_r1
W=${1:-2}
mkdir -p logs/fl_r1 logs/fl_fixsplit results/fl_fixsplit data/fl_ckpt_fixsplit
.venv/bin/python $R/make_jobs_r1.py
rm -f logs/fl_r1/R1_QUEUE_DONE
nohup bash -c "
  .venv/bin/python fl_study/run_grid.py --jobs $R/jobs_r1_extended.json --workers $W > logs/fl_r1/grid_extended.log 2>&1
  .venv/bin/python fl_study/run_grid.py --jobs $R/jobs_r1_fedbn_loso.json --workers $W > logs/fl_r1/grid_fedbn_loso.log 2>&1
  FL_RESULTS_DIR=/home/holiday/lung_ct/results/fl_fixsplit FL_CKPT_DIR=/home/holiday/lung_ct/data/fl_ckpt_fixsplit FL_LOGS_DIR=/home/holiday/lung_ct/logs/fl_fixsplit \
    .venv/bin/python fl_study/run_grid.py --jobs $R/jobs_r1_fixsplit.json --workers $W > logs/fl_r1/grid_fixsplit.log 2>&1
  .venv/bin/python fl_study/run_grid.py --jobs $R/jobs_r1_fedprox.json --workers $W > logs/fl_r1/grid_fedprox.log 2>&1
  date > logs/fl_r1/R1_QUEUE_DONE
" > /dev/null 2>&1 < /dev/null &
echo "launched revision-1 queue with $W workers; logs in logs/fl_r1/"
