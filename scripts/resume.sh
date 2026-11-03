#!/usr/bin/env bash
# One command to (re)start everything after a shutdown:
#   bash scripts/resume.sh
# Safe to run more than once: finished cross-validation folds are skipped.
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/Scripts/python}
C="--config configs/pretrained.yaml"

# 1) pretrained cascade: finish the flat baseline and run the test evaluation if not done yet
if [ ! -f runs_pretrained/results.json ]; then
  for n in triage family electrical environmental structural; do
    [ -f runs_pretrained/$n/best.pt ] || $PY -u -m solarnet.train --node $n $C
  done
  $PY -u -m solarnet.train --node flat $C
  $PY -u -m solarnet.evaluate --flat $C
fi
echo PRETRAINED_DONE >> pretrained_run.log

# 2) 10-fold cross-validation in two streams (skips finished folds), then aggregate
( $PY -u scripts/cv_cascade.py $C --folds 0 2 4 6 8 ) >> cv_streamA.log 2>&1 &
( $PY -u scripts/cv_cascade.py $C --folds 1 3 5 7 9 ) >> cv_streamB.log 2>&1 &
wait
$PY scripts/cv_cascade.py --aggregate
echo CV_DONE >> cv_run.log

# 3) push the results
bash scripts/auto_push.sh
