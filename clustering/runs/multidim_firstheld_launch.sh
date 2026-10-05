#!/bin/zsh
# The multidimensional theta fit (AR(1) + "spike" + Gompertz-Makeham mortality) with each
# person's FIRST row held out, people with at least four rows, for the type-from-an-initial-value
# exercise (descriptives/43_paper_type_from_initial.py). Detached, 4 chains x 3 threads at nice 10.
# Safe to relaunch: a finished fit is skipped.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=artifacts/multidim-health/theta-ssm-mort-firstho
mkdir -p $OUT
if [ -f $OUT/run_summary.json ]; then echo "already finished"; exit 0; fi
echo "[$(date +%H:%M:%S)] START theta-ssm-mort-firstho" >> artifacts/multidim-health/queue_firstheld.log
nice -n 10 .venv/bin/python clustering/run_multidim.py --variant ssm-firstho --physical theta --with-mortality \
  --holdout-last-k 1 --min-obs 4 --chains 4 --parallel-chains 4 --threads-per-chain 3 \
  --warmup 1500 --sampling 1000 --output $OUT > $OUT/run.log 2>&1
rc=$?
echo "[$(date +%H:%M:%S)] $([ $rc -eq 0 ] && echo DONE || echo FAILED rc=$rc) theta-ssm-mort-firstho" >> artifacts/multidim-health/queue_firstheld.log
echo "FIRSTHELD MULTIDIM COMPLETE" >> artifacts/multidim-health/queue_firstheld.log
