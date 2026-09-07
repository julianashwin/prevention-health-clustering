#!/bin/zsh
# Wait for the ssm2 queue to drain, then start the multidim batch.
# Chained rather than launched now so the two batches do not contend.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
while pgrep -f ssm_holdout_queue.py >/dev/null 2>&1; do sleep 120; done
echo "[$(date '+%H:%M:%S')] ssm2 queue drained; starting multidim batch"
exec .venv/bin/python clustering/runs/multidim_ssm_queue.py \
  --workers 2 --chains 4 --warmup 1500 --sampling 1000 --threads-per-chain 3
