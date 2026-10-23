# Cascaded-SolarNet

**Cascaded Convolutional Neural Network for Solar Panel Defect Classification**

Divyanshi Jain (RA2311026010639) · Syed Mohammad Ghouse (RA2311026010643) · Purvee Singh (RA2311026010645)

Flat single-stage CNNs on drone infrared imagery plateau around 93% accuracy on the
12-class InfraredSolarModules benchmark, mostly because of confusion within fault families
and class imbalance. Cascaded-SolarNet breaks the problem into three easier decisions
made by five small, specialised CNNs, using a single IR camera only.

```
                     ┌────────────────────────────┐
  IR image (24×40) ─▶│ Stage 1 · CNN-1  Triage    │── No-Anomaly ──▶ exit early
                     └────────────┬───────────────┘
                                  │ Defect
                     ┌────────────▼───────────────┐
                     │ Stage 2 · CNN-2  Family    │
                     └──┬──────────┬───────────┬──┘
              Electrical│ Environmental        │Structural
             ┌──────────▼┐ ┌───────▼──────┐ ┌──▼─────────┐
   Stage 3   │  CNN-3a   │ │   CNN-3b     │ │  CNN-3c    │
             └───────────┘ └──────────────┘ └────────────┘
                         final defect class (11 faults)
```

| Node | CNN | Blocks (Conv3×3-BN-ReLU-MaxPool) | Head |
|---|---|---|---|
| `triage` | CNN-1 | 32 → 64 | GAP + FC(2) |
| `family` | CNN-2 | 64 → 128 → 64, dropout 0.3 | GAP + FC(3) |
| `electrical` / `environmental` / `structural` | CNN-3a/b/c | 128 → 256 → 128, dropout 0.4 | GAP + FC(k) |
| `flat` (baseline) | — | 128 → 256 → 128, dropout 0.4 | GAP + FC(12) |

Widths follow the project poster. Change them in `configs/default.yaml`.

## Class hierarchy

The hierarchy lives in [`configs/taxonomy.yaml`](configs/taxonomy.yaml). The labels have to
match the dataset's `anomaly_class` strings exactly. The default grouping is:

| Family | Classes |
|---|---|
| — (Stage 1 exit) | No-Anomaly |
| Electrical | Cell, Cell-Multi, Diode, Diode-Multi, Hot-Spot, Hot-Spot-Multi |
| Environmental | Soiling, Shadowing, Vegetation |
| Structural | Cracking, Offline-Module |

> **Note on the poster:** the poster's leaf names (e.g. *Bypass Diode, Water Stain,
> Delamination, Ribbon*) are not labels in InfraredSolarModules, and its family sizes
> (3 + 4 + 5) add up to 12 defects instead of the dataset's 11. The architecture is
> implemented as on the poster. The classes are the dataset's real ones. To regroup
> them, edit `taxonomy.yaml`. No code changes are needed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

bash scripts/download_data.sh          # -> data/InfraredSolarModules/
python scripts/make_splits.py          # stratified 70/15/15, seed 42 -> data/splits.json
```

## Run on Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/techwack/cascaded-cnn/blob/main/notebooks/colab_train.ipynb)

[`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb) clones the repo, runs the tests,
downloads the dataset from the original GitHub repo, trains every node plus the baselines,
and evaluates the full cascade. Use a T4 GPU runtime.

## Train

Every node trains independently on ground-truth routing. For example, the Electrical CNN
only ever sees Electrical images.

```bash
python -m solarnet.train --node all        # triage, family, electrical, environmental, structural
python -m solarnet.train --node flat       # single-stage 12-class CNN baseline
python baselines/random_forest.py          # classical baseline
```

Checkpoints are saved to `runs/<node>/best.pt`, with per-epoch logs in `runs/<node>/history.json`.

## Evaluate

```bash
python -m solarnet.evaluate --flat
```

This runs the **whole cascade end to end** on the held-out test split. Routing errors
therefore count against the cascade. Results go to `runs/results.json`:

- accuracy, macro-F1, Cohen's kappa, per-class precision/recall/F1, 12×12 confusion matrix
- **per-stage error attribution**: missed defects at Stage 1, false alarms at Stage 1,
  wrong family at Stage 2, wrong subtype at Stage 3
- early-exit rate, parameter count, batch-1 CPU latency (ms/image)

## Preprocessing (from the dataset report)

| Step | Implementation |
|---|---|
| Normalise | ÷255, then z-score using **training-set** mean/std (`data/splits.json`) |
| Augment (train only) | h/v flips p=0.5, brightness ±20%, contrast ±15%, Gaussian noise σ=0.02 |
| Imbalance | weighted CE, `w_r = N / (K · n_r)` per node; early exit at Stage 1 |
| Split | stratified 70/15/15, seed 42, shared by every model |
| Upsampling | optional, `data.upsample: 2` → 80×48 |

90° rotations are left out because the images are 40×24, so a quarter turn would change
the input shape. Flipping both ways already gives the 180° rotation.

## Experiment plan

1. Baselines: random forest, flat CNN (`--node flat`), and published EfficientNet+SVM numbers.
2. Cascade vs flat with the same backbone widths: accuracy, macro-F1, kappa.
3. Ablations: no early exit (send everything to Stage 2), unweighted loss, upsample 1 vs 2.
4. Error propagation: per-stage attribution from `results.json`.
5. Latency/params for edge deployment.

Known limitation: the benchmark has no module or site IDs, so near-duplicate crops of the
same module can fall into both train and test.

## Layout

```
configs/      default.yaml (hyper-parameters), taxonomy.yaml (class hierarchy)
scripts/      download_data.sh, make_splits.py
src/solarnet/ taxonomy.py, data.py, models.py, common.py, train.py, evaluate.py
baselines/    random_forest.py
tests/        pytest suite (synthetic data, no download needed)
```

## Reference

Millendorf, Obropta, Vadhavkar. *Infrared Solar Module Dataset for Anomaly Detection.*
ICLR 2020 Workshop on AI for Earth Sciences.
