#!/usr/bin/env bash
# Download InfraredSolarModules (Millendorf et al., ICLR 2020 workshop; CC BY 4.0)
# Needs a Kaggle API token in ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API).
set -euo pipefail
mkdir -p data
kaggle datasets download -d rimmelasghar/solar-anamolies-dataset -p data --unzip
# Find wherever module_metadata.json landed and expose it at data/InfraredSolarModules
meta=$(find data -name module_metadata.json | head -1)
if [ -z "$meta" ]; then echo "module_metadata.json not found under data/" >&2; exit 1; fi
src=$(dirname "$meta")
if [ "$src" != "data/InfraredSolarModules" ]; then
  rm -rf data/InfraredSolarModules && mv "$src" data/InfraredSolarModules
fi
echo "dataset ready at data/InfraredSolarModules"
