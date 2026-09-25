#!/bin/zsh
# The next batch on the health contract, detached, one fit at a time,
# 4 chains x 3 threads = 12 cores at nice 10.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-next
exec nice -n 10 .venv/bin/python clustering/runs/health_next_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  > artifacts/health-next/queue.log 2>&1
