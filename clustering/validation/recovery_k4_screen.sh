#!/bin/zsh
# K=4 state-space recovery in a detached screen session; 4 chains x 1 thread.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/recovery-ssm-k4
exec .venv/bin/python clustering/validation/synthetic_recovery_ssm_k4.py \
  --persons 5000 --warmup 1000 --sampling 600 --chains 4 \
  > artifacts/recovery-ssm-k4/run.log 2>&1
