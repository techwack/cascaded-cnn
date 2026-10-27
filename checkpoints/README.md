# Shared checkpoints

`stage3_improved/` holds the three trained Stage 3 specialists (electrical, environmental,
structural) from `configs/improved.yaml`. `configs/improved3.yaml` reuses them unchanged.

Validation macro-F1: electrical 0.786, environmental 0.822, structural 0.955.
Trained on the 70% train split (seed 42), taxonomy `configs/taxonomy.yaml`.

To use them for the `improved3` cascade:

```bash
git pull
mkdir -p runs_improved3
cp -r checkpoints/stage3_improved/* runs_improved3/
python -m solarnet.train --node triage --config configs/improved3.yaml
python -m solarnet.train --node family --config configs/improved3.yaml
python -m solarnet.evaluate --config configs/improved3.yaml
```

These only match `taxonomy.yaml` (Electrical / Environmental / Structural), not the
data-driven grouping in `taxonomy_data.yaml`.
