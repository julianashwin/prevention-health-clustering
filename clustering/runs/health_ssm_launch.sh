#!/bin/zsh
# The three AR(1) plus measurement error K=3 fits on the paper's measure, detached, one at a
# time, 4 chains x 3 threads = 12 cores at nice 10.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-ssm
exec nice -n 10 .venv/bin/python clustering/runs/health_ssm_queue.py --workers 1 \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 \
  > artifacts/health-ssm/queue.log 2>&1
