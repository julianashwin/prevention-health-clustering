#!/bin/zsh
# K=4 state-space fits in a detached screen session, alongside the multidim
# batch. One fit at a time, 4 chains x 1 thread = 4 cores.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python clustering/runs/k4_queue.py \
  --chains 4 --parallel-chains 4 --warmup 1500 --sampling 1000 \
  --threads-per-chain 1
