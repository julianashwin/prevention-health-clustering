#!/bin/zsh
# The three AR(1) plus measurement error + birth-decade-cohort fits, queued behind the
# AR(1) plus measurement error queue: waits for its QUEUE COMPLETE line, then runs one at a
# time, 4 chains x 3 threads at nice 10.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-ssm-cohort
until grep -q "QUEUE COMPLETE" artifacts/health-ssm/queue.log 2>/dev/null; do sleep 120; done
exec nice -n 10 .venv/bin/python clustering/runs/health_cohort_queue.py --workers 1 \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  > artifacts/health-ssm-cohort/queue.log 2>&1
