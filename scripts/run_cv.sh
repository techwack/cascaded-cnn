#!/usr/bin/env bash
# 10-fold cross-validation of the pretrained cascade in two parallel streams, then aggregate.
# Waits for the pretrained run (run_pretrained.sh) to finish first if its log shows it is still going.
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/Scripts/python}
if [ -f pretrained_run.log ] && ! grep -q PRETRAINED_DONE pretrained_run.log; then
  until grep -q PRETRAINED_DONE pretrained_run.log; do sleep 20; done
fi
C="--config configs/pretrained.yaml"
( $PY -u scripts/cv_cascade.py $C --folds 0 2 4 6 8 ) > cv_streamA.log 2>&1 &
( $PY -u scripts/cv_cascade.py $C --folds 1 3 5 7 9 ) > cv_streamB.log 2>&1 &
wait
$PY scripts/cv_cascade.py --aggregate
echo CV_DONE
