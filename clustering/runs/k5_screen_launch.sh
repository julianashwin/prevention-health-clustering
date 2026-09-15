#!/bin/zsh
# K=5 P-FUNC under the state-space specification, in a detached screen session.
# 4 chains x 1 thread = 4 cores, at nice 10 so interactive work takes precedence.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=artifacts/k5/physfunc-ssm-k5
mkdir -p "$OUT"
exec nice -n 10 .venv/bin/python clustering/run_fit.py \
  --model physgrm-func-ssm-k5 \
  --contract "$PWD/data/processed/contracts/physfunc_lifecycle_20_89_minobs3_v1" \
  --output "$OUT" \
  --chains 4 --parallel-chains 4 --threads-per-chain 1 \
  --warmup 1500 --sampling 1000 \
  > "$OUT/run.log" 2>&1
