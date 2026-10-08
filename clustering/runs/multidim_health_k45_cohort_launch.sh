#!/bin/zsh
# Multidimensional theta K = 4 and K = 5 with birth-decade shifts (AR(1) + "spike" + mortality,
# --cohort decade), queued behind the no-shift K = 5 fit: waits for theta-ssm-mort-k5's
# run_summary.json, then runs the pair one at a time, 4 chains x 3 threads at nice 10.
# Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=artifacts/multidim-health/queue_k45_cohort.log
echo "[$(date '+%d %b %H:%M:%S')] WAITING for theta-ssm-mort-k5" >> $LOG
until [ -f artifacts/multidim-health/theta-ssm-mort-k5/run_summary.json ]; do sleep 600; done
exec nice -n 10 .venv/bin/python clustering/runs/multidim_health_k45_queue.py --cohort \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> $LOG 2>&1
