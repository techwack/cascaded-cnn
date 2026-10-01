"""Create the stratified 70/15/15 split and training-set normalisation stats.

    python scripts/make_splits.py

Every model (cascade nodes and baselines) reads data/splits.json, so all
results are on the same held-out test set.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from solarnet.data import load_image, load_records, make_splits  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402

import yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tax = Taxonomy.load(args.taxonomy)
    dc = cfg["data"]

    records = load_records(dc["root"])
    counts = Counter(r["label"] for r in records)
    unknown = set(counts) - set(tax.classes)
    missing = set(tax.classes) - set(counts)
    if unknown or missing:
        sys.exit(f"taxonomy.yaml does not match the dataset.\n  in data, not taxonomy: {sorted(unknown)}"
                 f"\n  in taxonomy, not data: {sorted(missing)}")

    splits = make_splits(records, dc["val_frac"], dc["test_frac"], dc["seed"])
    by_id = {r["id"]: r for r in records}
    train_px = np.stack([load_image(by_id[i]["path"]) for i in splits["train"]])
    stats = {"mean": float(train_px.mean()), "std": float(train_px.std())}

    out = Path(dc["splits_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({**splits, "stats": stats, "seed": dc["seed"]}, open(out, "w"))

    print(f"{len(records)} images -> " + ", ".join(f"{k} {len(v)}" for k, v in splits.items()))
    print(f"train mean {stats['mean']:.4f} std {stats['std']:.4f}")
    print(f"{'class':16s} {'total':>6s} {'train':>6s} {'val':>5s} {'test':>5s}")
    split_counts = {k: Counter(by_id[i]["label"] for i in v) for k, v in splits.items()}
    for c in tax.classes:
        print(f"{c:16s} {counts[c]:6d} {split_counts['train'][c]:6d} {split_counts['val'][c]:5d} {split_counts['test'][c]:5d}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
