#!/bin/zsh
# Multidimensional fits (physical + mental + Gompertz-Makeham mortality) on the
# paper's measure, detached, one fit at a time, 4 chains x 3 threads = 12 of
# the 14 cores at nice 10. Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/multidim-health
exec nice -n 10 .venv/bin/python clustering/runs/multidim_health_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> artifacts/multidim-health/queue.log 2>&1
