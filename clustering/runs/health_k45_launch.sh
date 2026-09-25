#!/bin/zsh
# K = 4 and K = 5 on the health measure, detached, one fit at a time,
# 4 chains x 3 threads = 12 cores at nice 10. Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-k45
exec nice -n 10 .venv/bin/python clustering/runs/health_k45_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> artifacts/health-k45/queue.log 2>&1
