#!/usr/bin/env bash
# Train the cascade on the data-driven family grouping (triage is reused if already trained).
cd "$(dirname "$0")/.."
PY=${PY:-./.venv/Scripts/python}
C="--config configs/datagroup.yaml --taxonomy configs/taxonomy_data.yaml"
[ -f runs_datagroup/triage/best.pt ] || $PY -u -m solarnet.train --node triage $C
for n in family groupa groupb groupc; do $PY -u -m solarnet.train --node $n $C; done
$PY -u -m solarnet.evaluate $C
echo DATAGROUP_DONE
