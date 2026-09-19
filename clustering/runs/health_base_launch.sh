#!/bin/zsh
# The three baseline K=3 fits on the paper's measure, detached, one at a time,
# 4 chains x 2 threads = 8 cores at nice 10 (the K=5 fit holds 4 more).
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-base
exec nice -n 10 .venv/bin/python clustering/runs/health_base_queue.py --workers 1 \
  --chains 4 --parallel-chains 4 --threads-per-chain 2 --warmup 1000 --sampling 1000 \
  > artifacts/health-base/queue.log 2>&1
