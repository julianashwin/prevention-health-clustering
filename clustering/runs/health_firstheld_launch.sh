#!/bin/zsh
# The two K=3 fits on theta with each person's first row held out (independent, AR(1) + "spike"),
# queued behind the mental-health queue: waits for its MENTAL QUEUE COMPLETE line, then runs
# one at a time, 4 chains x 3 threads = 12 of 14 cores at nice 10. Safe to relaunch.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-firstheld
until grep -q "MENTAL QUEUE COMPLETE" artifacts/mental-health/queue.log 2>/dev/null; do sleep 300; done
exec nice -n 10 .venv/bin/python clustering/runs/health_firstheld_queue.py --workers 1 \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  >> artifacts/health-firstheld/queue.log 2>&1
