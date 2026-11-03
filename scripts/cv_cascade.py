"""10-fold stratified cross-validation of the cascade (same protocol family as Duranay 2023, but leak-free).

    python scripts/cv_cascade.py --config configs/pretrained.yaml --folds 0 2 4 6 8
    python scripts/cv_cascade.py --config configs/pretrained.yaml --folds 1 3 5 7 9
    python scripts/cv_cascade.py --aggregate

Fold k: images of fold k are the test set. From the other nine folds, 1/9 (stratified) is held out as a
validation set for early stopping and the rest trains the five cascade CNNs from scratch/pretrained weights.
Nothing from the test fold is used for training, early stopping, normalisation or model selection.
Per-fold metrics go to runs_cv/fold{k}/metrics.json; --aggregate writes results/cv_pretrained.json.
"""
import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarnet.common import device, load_cascade, load_config, preprocess, to_batch  # noqa: E402
from solarnet.data import SplitData, load_image, load_records  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402
from solarnet.train import train_node  # noqa: E402

CV_DIR = ROOT / "runs_cv"


def subset(records, images, idx) -> SplitData:
    sd = SplitData.__new__(SplitData)
    sd.records = [records[i] for i in idx]
    sd.images = images[idx]
    sd.labels = [records[i]["label"] for i in idx]
    return sd


def run_fold(k, cfg, tax, records, images, labels, folds):
    train_val, test = folds[k]
    tr, va = train_test_split(train_val, test_size=1 / 9, stratify=labels[train_val], random_state=cfg["data"]["seed"])
    splits = {"train": subset(records, images, tr), "val": subset(records, images, va), "test": subset(records, images, test)}
    stats = {"mean": float(images[tr].mean()), "std": float(images[tr].std())}
    c = copy.deepcopy(cfg)
    out_dir = CV_DIR / f"fold{k}"
    c["train"]["out_dir"] = str(out_dir)
    print(f"=== fold {k}: train {len(tr)} val {len(va)} test {len(test)}", flush=True)
    for node in tax.nodes:
        train_node(c, tax, node, splits, stats)

    dev = device()
    cascade = load_cascade(c, tax, out_dir).to(dev).eval()
    up = c["data"]["upsample"]
    x = preprocess(to_batch(images[test]), stats, up)
    pred = torch.cat([cascade.predict(x[i:i + 250].to(dev))["pred"].cpu() for i in range(0, len(x), 250)]).numpy()
    y = np.array([tax.index(records[i]["label"]) for i in test])
    healthy = tax.index(tax.healthy)
    m = {"fold": k, "n_test": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
         "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)), "kappa": float(cohen_kappa_score(y, pred)),
         "binary_accuracy": float(((pred != healthy) == (y != healthy)).mean())}
    print(json.dumps(m), flush=True)
    (out_dir / "metrics.json").write_text(json.dumps(m, indent=1))
    for pt in out_dir.rglob("best.pt"):  # keep disk use small
        pt.unlink()


def aggregate():
    ms = [json.load(open(p)) for p in sorted(CV_DIR.glob("fold*/metrics.json"))]
    out = {"n_folds": len(ms), "folds": ms}
    for key in ("accuracy", "macro_f1", "kappa", "binary_accuracy"):
        v = np.array([m[key] for m in ms])
        out[key + "_mean"], out[key + "_std"] = float(v.mean()), float(v.std())
    Path(ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "cv_pretrained.json").write_text(json.dumps(out, indent=1))
    print(f"{len(ms)} folds: accuracy {out['accuracy_mean']:.4f} +- {out['accuracy_std']:.4f}, macro-F1 {out['macro_f1_mean']:.4f}, "
          f"binary {out['binary_accuracy_mean']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pretrained.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--folds", type=int, nargs="*", default=[])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        return aggregate()

    cfg, tax = load_config(args.config), Taxonomy.load(args.taxonomy)
    recs = load_records(cfg["data"]["root"])
    images = np.stack([load_image(r["path"]) for r in recs])
    labels = np.array([r["label"] for r in recs])
    folds = list(StratifiedKFold(args.k, shuffle=True, random_state=cfg["data"]["seed"]).split(images, labels))
    for k in args.folds:
        if (CV_DIR / f"fold{k}" / "metrics.json").exists():
            print(f"fold {k} already done, skipping")
            continue
        run_fold(k, cfg, tax, recs, images, labels, folds)


if __name__ == "__main__":
    main()
