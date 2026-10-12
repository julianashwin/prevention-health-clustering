#!/bin/zsh
# K = 4 and K = 5 AR(1) + "spike" on the frailty index and log frailty, detached, one fit at a
# time, 4 chains x 3 threads at nice 10 (launched 12 Oct alongside the cohort K = 4/5 queue).
# Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-k45
exec nice -n 10 .venv/bin/python clustering/runs/health_k45_queue.py --frailty \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> artifacts/health-k45/queue_frailty.log 2>&1
