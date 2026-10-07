#!/bin/zsh
# The multidimensional theta fit (AR(1) + "spike" + Gompertz-Makeham mortality) with each
# person's FIRST TWO rows held out, people with at least five rows: the two-row typology for the
# type-from-initial exercise (descriptives/43_paper_type_from_initial.py --first2), which uses the
# first-row fit as a placeholder until this lands. Waits for the windows queue to finish first
# (two fits at a time), then runs detached, 4 chains x 3 threads at nice 10. Safe to relaunch.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT=artifacts/multidim-health/theta-ssm-mort-first2ho
LOG=artifacts/multidim-health/queue_first2.log
mkdir -p $OUT
if [ -f $OUT/run_summary.json ]; then echo "already finished"; exit 0; fi
echo "[$(date '+%d %b %H:%M:%S')] WAITING for the windows queue" >> $LOG
until grep -q "WINDOWS QUEUE COMPLETE" artifacts/health-windows/queue.log 2>/dev/null; do sleep 600; done
echo "[$(date '+%d %b %H:%M:%S')] START theta-ssm-mort-first2ho" >> $LOG
nice -n 10 .venv/bin/python clustering/run_multidim.py --variant ssm-firstho --physical theta --with-mortality \
  --holdout-last-k 2 --min-obs 5 --chains 4 --parallel-chains 4 --threads-per-chain 3 \
  --warmup 1500 --sampling 1000 --output $OUT > $OUT/run.log 2>&1
rc=$?
echo "[$(date '+%d %b %H:%M:%S')] $([ $rc -eq 0 ] && echo DONE || echo FAILED rc=$rc) theta-ssm-mort-first2ho" >> $LOG
echo "FIRST2 MULTIDIM COMPLETE" >> $LOG
