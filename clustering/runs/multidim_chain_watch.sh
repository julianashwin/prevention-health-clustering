#!/bin/zsh
# Watch the h-ssm-mort fit. Chain 4 has written no draws after 22 h while
# chains 1-3 sample in one mode. Once chains 1, 2 and 3 each hold the planned
# 1,000 draws and chain 4 still has none, stop the CmdStan process so the
# queue moves on to the two baseline fits; the three complete chains are then
# a full posterior sample for the fit (salvaged from the csvs). If chain 4
# starts writing draws, do nothing and let the fit finish on its own.
cd /Users/julianashwin/Documents/GitHub/prevention-health-clustering
D=artifacts/multidim-health/h-ssm-mort/chains
LOG=artifacts/multidim-health/watch_h_ssm_mort.log
echo "[$(date '+%F %T')] watcher started" >> $LOG
while true; do
  PID=$(pgrep -f "mixture_multidim_panel id=" | head -n 1)
  if [ -z "$PID" ]; then echo "[$(date '+%F %T')] no CmdStan process; watcher exits" >> $LOG; exit 0; fi
  n=()
  for c in 1 2 3 4; do
    f=$(ls $D/*_${c}.csv 2>/dev/null | head -n 1)
    n[$c]=$([ -n "$f" ] && grep -vc '^#' "$f" || echo 0)
  done
  echo "[$(date '+%F %T')] draws ${n[1]} ${n[2]} ${n[3]} ${n[4]}" >> $LOG
  if [ ${n[1]} -ge 1001 ] && [ ${n[2]} -ge 1001 ] && [ ${n[3]} -ge 1001 ] && [ ${n[4]} -le 1 ]; then
    echo "[$(date '+%F %T')] chains 1-3 complete, chain 4 never sampled: stopping pid $PID" >> $LOG
    kill $PID
    exit 0
  fi
  sleep 600
done
