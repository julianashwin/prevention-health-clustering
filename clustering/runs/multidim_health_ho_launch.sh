#!/bin/zsh
# Held-out multidimensional fits (AR(1) + "spike" + mortality, theta then h),
# detached, one at a time, 4 chains x 3 threads = 12 of 14 cores at nice 10.
# Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/multidim-health
exec nice -n 10 .venv/bin/python clustering/runs/multidim_health_ho_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> artifacts/multidim-health/queue_ho.log 2>&1
