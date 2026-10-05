#!/bin/zsh
# The theta and h AR(1) + "spike" + birth-decade-shift fits with the first two rows held out,
# run ALONGSIDE the frailty/first-held queue (each fit uses 4-6 cores of 14 at 12 threads,
# so two queues together sit near 80%). Safe to relaunch.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-firstheld
exec nice -n 10 .venv/bin/python clustering/runs/cohort_first2_queue.py \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  >> artifacts/health-firstheld/queue_cohort_first2.log 2>&1
