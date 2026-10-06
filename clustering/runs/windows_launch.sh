#!/bin/zsh
# theta AR(1) + "spike" refitted on the ten age windows, one at a time, 4 chains x 3 threads at
# nice 10, run alongside the frailty/first-held queue. Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-windows
exec nice -n 10 .venv/bin/python clustering/runs/windows_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  >> artifacts/health-windows/queue.log 2>&1
