#!/bin/bash
# MOPS 監控迴圈：每 5 分鐘 poll；每天 00:20 後對帳前一天。停止：kill $(cat ~/mops-monitor/run.pid)
export TZ=Asia/Taipei MOPS_DB="$HOME/mops-monitor/data/mops.sqlite"
cd "$HOME/mops-monitor" || exit 1
if [ -f run.pid ] && kill -0 "$(cat run.pid)" 2>/dev/null; then echo "已在執行"; exit 0; fi
echo $$ > run.pid
PY=/usr/local/bin/python3
while true; do
  "$PY" mops_monitor.py poll
  today=$(date +%Y%m%d)
  if [ "$(date +%H%M)" -ge 0020 ] && [ "$(cat .reconciled 2>/dev/null)" != "$today" ]; then
    "$PY" mops_monitor.py reconcile --date yesterday && echo "$today" > .reconciled
  fi
  /bin/sleep $((300 - $(date +%s) % 300))
done
