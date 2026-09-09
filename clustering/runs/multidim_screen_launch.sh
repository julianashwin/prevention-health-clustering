#!/bin/zsh
# Run the multidim batch inside a detached screen session so it survives the
# terminal going away. One fit at a time, 4 chains x 1 thread = 4 cores, well
# under half of this machine's 14.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
# Belt and braces: cap any library-level threading too, so the only
# parallelism is the four chains.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec .venv/bin/python clustering/runs/multidim_ssm_queue.py \
  --workers 1 --chains 4 --warmup 1500 --sampling 1000 --threads-per-chain 1
