#!/bin/zsh
# The three univariate K=3 fits on the mental GRM (independent, AR(1) + "spike", + cohort),
# queued behind the multidimensional cohort queue: waits for its COHORT QUEUE COMPLETE line,
# then runs one at a time, 4 chains x 3 threads = 12 of 14 cores at nice 10.
# Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/mental-health
until grep -q "COHORT QUEUE COMPLETE" artifacts/multidim-health/queue_cohort.log 2>/dev/null; do sleep 300; done
exec nice -n 10 .venv/bin/python clustering/runs/mental_queue.py --workers 1 \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  >> artifacts/mental-health/queue.log 2>&1
