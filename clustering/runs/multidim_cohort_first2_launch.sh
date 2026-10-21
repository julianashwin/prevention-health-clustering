#!/bin/zsh
# The multidimensional theta fit with birth-decade shifts (AR(1) + "spike" + Gompertz-Makeham
# mortality, --cohort decade, as theta-ssm-mort-cohort) with each person's FIRST TWO rows held
# out, people with at least five rows: the cohort twin of theta-ssm-mort-first2ho for the
# type-from-initial exercise. Runs detached, 4 chains x 3 threads at nice 10, alongside one other
# fit (launched 19 Oct next to the frailty K = 5 queue). Safe to relaunch.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=artifacts/multidim-health/theta-ssm-mort-cohort-first2ho
LOG=artifacts/multidim-health/queue_cohort_first2.log
mkdir -p $OUT
if [ -f $OUT/run_summary.json ]; then echo "already finished"; exit 0; fi
echo "[$(date '+%d %b %H:%M:%S')] START theta-ssm-mort-cohort-first2ho" >> $LOG
nice -n 10 .venv/bin/python clustering/run_multidim.py --variant ssm-firstho --physical theta --with-mortality \
  --cohort decade --holdout-last-k 2 --min-obs 5 --chains 4 --parallel-chains 4 --threads-per-chain 3 \
  --warmup 1500 --sampling 1000 --output $OUT > $OUT/run.log 2>&1
rc=$?
echo "[$(date '+%d %b %H:%M:%S')] $([ $rc -eq 0 ] && echo DONE || echo FAILED rc=$rc) theta-ssm-mort-cohort-first2ho" >> $LOG
echo "COHORT FIRST2 MULTIDIM COMPLETE" >> $LOG
