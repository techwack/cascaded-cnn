#!/usr/bin/env bash
# Train the pretrained (EfficientNet-B0) cascade and flat baseline, then evaluate both.
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/Scripts/python}
C="--config configs/pretrained.yaml"
$PY -u -m solarnet.train --node all $C
$PY -u -m solarnet.train --node flat $C
$PY -u -m solarnet.evaluate --flat $C
echo PRETRAINED_DONE
