#!/bin/zsh
# Multidimensional theta K = 4 and K = 5 (AR(1) + "spike" + mortality), queued behind the
# frailty/first-held queue: waits for its ROUND2 QUEUE COMPLETE line, then runs one at a
# time, 4 chains x 3 threads at nice 10. Safe to relaunch: finished fits are skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
until grep -q "ROUND2 QUEUE COMPLETE" artifacts/health-frailty/queue.log 2>/dev/null; do sleep 600; done
exec nice -n 10 .venv/bin/python clustering/runs/multidim_health_k45_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> artifacts/multidim-health/queue_k45.log 2>&1
