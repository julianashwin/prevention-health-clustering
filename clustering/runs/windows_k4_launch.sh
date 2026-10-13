#!/bin/zsh
# theta AR(1) + "spike" at K = 4 refitted on the ten age windows (the K = 4 twin of the windows
# figure), one at a time, 4 chains x 3 threads at nice 10. Waits for the log-frailty K = 4 fit to
# finish so that two fits run at a time (the frailty / log-frailty K = 5 pair is deferred behind
# this: relaunch runs/health_k45_frailty_launch.sh afterwards, it skips the finished fits).
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=artifacts/health-windows/queue_k4.log
echo "[$(date '+%d %b %H:%M:%S')] WAITING for health-logfrailty-ssm-k4" >> $LOG
until [ -f artifacts/health-k45/health-logfrailty-ssm-k4/run_summary.json ]; do sleep 300; done
exec nice -n 10 .venv/bin/python clustering/runs/windows_queue.py --k 4 \
  --chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1500 --sampling 1000 \
  >> $LOG 2>&1
