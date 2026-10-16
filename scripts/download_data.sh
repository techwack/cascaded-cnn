#!/usr/bin/env bash
# Download InfraredSolarModules (Millendorf et al., ICLR 2020 workshop; CC BY 4.0)
# from the original RaptorMaps GitHub repo. No account or token needed.
set -euo pipefail
mkdir -p data
curl -L --fail -o data/InfraredSolarModules.zip \
  https://github.com/RaptorMaps/InfraredSolarModules/raw/master/2020-02-14_InfraredSolarModules.zip
rm -rf data/InfraredSolarModules
unzip -q -o data/InfraredSolarModules.zip -x "__MACOSX/*" -d data
test -f data/InfraredSolarModules/module_metadata.json || { echo "module_metadata.json not found" >&2; exit 1; }
echo "dataset ready at data/InfraredSolarModules ($(ls data/InfraredSolarModules/images | wc -l) images)"
