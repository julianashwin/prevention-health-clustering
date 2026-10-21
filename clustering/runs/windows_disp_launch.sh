#!/bin/zsh
# K = 3 theta windows refitted from dispersed inits (build_inits, per-chain jitter) instead of the
# shared k-means partition, to check the windows' classes do not depend on a common start.
# Two queues side by side (sliding, backward), 4 chains x 3 threads each at nice 10, the same
# warmup and sampling as windows_launch.sh. Tags theta-ssm-disp-{lo}-{hi}. Safe to relaunch.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p artifacts/health-windows
SET=(--chains 4 --parallel-chains 4 --threads-per-chain 3 --warmup 1000 --sampling 1000 --init dispersed --init-seed 2026)
nohup nice -n 10 .venv/bin/python clustering/runs/windows_queue.py $SET --windows 20-40,30-50,40-60,50-70,60-80,70-90 \
  >> artifacts/health-windows/queue_disp_sliding.log 2>&1 &
nohup nice -n 10 .venv/bin/python clustering/runs/windows_queue.py $SET --windows 60-90,50-90,40-90,30-90 \
  >> artifacts/health-windows/queue_disp_backward.log 2>&1 &
