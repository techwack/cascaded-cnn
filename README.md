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

## Results (test split, 3,000 images, seed 42, single run)

| Config | Model | Accuracy | Macro-F1 | Kappa | Stage-1 binary acc |
|---|---|---|---|---|---|
| `default.yaml` (brightness/contrast/noise jitter) | Cascade | 0.617 | 0.368 | 0.461 | 0.810 |
| | Flat CNN | 0.476 | 0.379 | 0.357 | n/a |
| `no_photometric.yaml` (flips only) | Cascade | 0.711 | 0.509 | 0.583 | 0.860 |
| | Flat CNN | 0.685 | 0.571 | 0.586 | n/a |
| `improved.yaml` (80x48, 2 convs/block, no jitter) | Cascade | 0.760 | 0.582 | 0.656 | 0.895 |
| | Flat CNN | 0.768 | 0.653 | 0.689 | n/a |
| | Cascade + flip TTA | 0.764 | 0.582 | 0.660 | n/a |
| `improved2.yaml` (improved + sqrt-balanced sampler + label smoothing) | Cascade | 0.755 | 0.576 | 0.648 | 0.891 |
| **`improved3.yaml`** (improved + larger Stage 1/2 CNNs) | **Cascade** | **0.796** | **0.639** | **0.714** | **0.938** |

Raw metrics are in [`results/`](results/). Notes:

- Photometric jitter hurt: thermal defects are defined by absolute brightness.
- With `no_photometric.yaml` the cascade beat the flat CNN on accuracy but not macro-F1. With `improved.yaml` the **flat CNN is slightly better on every accuracy metric** (and has 1.48M vs 4.87M params); the cascade's advantage is speed (55% of images exit at Stage 1). Tried without gain: triage-threshold tuning (val-optimal threshold was 0.50, identical result) and balanced sampling + label smoothing (`improved2.yaml`, slightly worse). Larger Stage 1/2 networks (`improved3.yaml`) helped most: Stage-1 missed defects fell 233 -> 108, and the cascade now beats the improved flat CNN on accuracy (0.796 vs 0.768) and kappa (0.714 vs 0.689) but not macro-F1 (0.639 vs 0.653). Stage 2 (wrong family, ~300 errors) is still the largest error source. Inference-only tweaks did not help on `improved.yaml`: soft routing 0.759 acc (vs 0.760 hard), soft + TTA 0.763, threshold tuning 0.760.
- Largest error source is Stage 2 (wrong family), then missed defects at Stage 1.
- The random forest has not been run locally (a scikit-learn DLL is blocked by Windows Application Control); run it on Colab. Latencies in `results/improved*.json` were measured while the GPU was busy training and are not reliable.
- 93% in the literature is not a 12-class figure we can compare against directly; see the report for the binary vs multi-class gap.

## Status and who is doing what

**Best so far:** `improved3.yaml`, cascade test accuracy 0.796, macro-F1 0.639, kappa 0.714, Stage-1 binary accuracy 0.938 (see Results). The 12-class result is far below the 93% quoted in the literature; that figure most likely refers to the binary healthy-vs-defect task.

**Where the errors are** (`improved3`, 613 errors on 3,000 test images): Stage 2 wrong family 300, Stage 3 wrong subtype 126, Stage 1 missed defects 108, Stage 1 false alarms 79. Stage 2 is the bottleneck.

| Area | Status |
|---|---|
| Stage 3 (electrical, environmental, structural) | Trained with `improved.yaml`; reused unchanged by `improved3.yaml`. Val macro-F1: 0.786 / 0.822 / 0.955. No need to retrain unless trying to improve Electrical. |
| Stage 1 and 2 (triage, family) | Purvee: experiments on larger/longer Stage 1 and 2 networks. |
| Data-driven family grouping | Running (`scripts/run_datagroup.sh`, `configs/datagroup.yaml`, `configs/taxonomy_data.yaml`); result not in yet. |
| Random forest baseline | Not run locally (blocked DLL on one machine); run it on Colab. |

Trained checkpoints (`*.pt`) and `runs*/` are git-ignored, so they are not in the repo. Ask for them if you need to evaluate the full cascade without retraining.

To evaluate the full cascade after training: `python -m solarnet.evaluate --config configs/improved3.yaml`

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
