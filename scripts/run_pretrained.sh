#!/usr/bin/env bash
# Train the pretrained (EfficientNet-B0) cascade and flat baseline in two parallel streams, then evaluate.
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/Scripts/python}
C="--config configs/pretrained.yaml"
( $PY -u -m solarnet.train --node triage $C; $PY -u -m solarnet.train --node family $C; $PY -u -m solarnet.train --node flat $C ) > pretrained_streamA.log 2>&1 &
( for n in electrical environmental structural; do $PY -u -m solarnet.train --node $n $C; done ) > pretrained_streamB.log 2>&1 &
wait
$PY -u -m solarnet.evaluate --flat $C
echo PRETRAINED_DONE
