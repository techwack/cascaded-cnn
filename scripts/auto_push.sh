#!/usr/bin/env bash
# Every 10 minutes: refresh results/LIVE_STATUS.md and push result files to GitHub (main) if they changed.
# Stops after the cross-validation has finished (CV_DONE in cv_run.log) and one final push.
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/Scripts/python}
push_once() {
  $PY scripts/live_status.py > /dev/null 2>&1
  git add results/LIVE_STATUS.md results/pretrained.json results/cv_pretrained.json 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "Auto-update: training progress and results" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
    git pull --rebase --autostash -q origin main 2>/dev/null
    git push -q origin main 2>/dev/null && echo "pushed $(date +%H:%M)"
  fi
}
while true; do
  push_once
  if [ -f cv_run.log ] && grep -q CV_DONE cv_run.log; then push_once; echo AUTOPUSH_DONE; break; fi
  sleep 600
done
