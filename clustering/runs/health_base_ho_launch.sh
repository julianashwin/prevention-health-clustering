#!/bin/zsh
# The two baseline held-out fits, detached, after the K = 4/5 queue has exited.
# 4 chains x 3 threads = 12 cores at nice 10. Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
while pgrep -f "health_k45_queue.py" > /dev/null; do sleep 300; done
exec nice -n 10 .venv/bin/python clustering/runs/health_base_ho_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  >> artifacts/health-next/queue_base_ho.log 2>&1
