#!/bin/zsh
# Frailty / log-frailty K=3 fits, then the h first-row and theta/h first-two-rows held-out
# fits, one at a time, 4 chains x 3 threads = 12 of 14 cores at nice 10. Safe to relaunch.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-frailty
exec nice -n 10 .venv/bin/python clustering/runs/frailty_firstheld_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  >> artifacts/health-frailty/queue.log 2>&1
