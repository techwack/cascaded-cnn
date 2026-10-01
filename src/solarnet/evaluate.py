"""End-to-end evaluation of the cascade (and optionally the flat baseline) on the test split.

    python -m solarnet.evaluate                 # cascade
    python -m solarnet.evaluate --flat          # also evaluate runs/flat/best.pt for comparison

Writes runs/results.json with accuracy, macro-F1, Cohen's kappa, per-class
report, 12x12 confusion matrix, per-stage error attribution, parameter
counts and batch-1 latency.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score

from .common import load_cascade, load_config, load_node, preprocess, to_batch
from .data import load_splits
from .taxonomy import Taxonomy


def metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y, pred),
        "per_class": classification_report(y, pred, labels=labels, target_names=classes,
                                           output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def stage_errors(y: np.ndarray, out: dict, tax: Taxonomy) -> dict:
    """Attribute every cascade mistake to the first stage that went wrong."""
    healthy = tax.index(tax.healthy)
    true_defect = y != healthy
    true_fam = np.array([tax.family_names.index(tax.family_of(tax.classes[c])) if c != healthy else -1 for c in y])
    is_defect = out["is_defect"].numpy()
    fam = out["family"].numpy()
    wrong = out["pred"].numpy() != y

    triage_miss = true_defect & ~is_defect           # defect exited early as healthy
    triage_false_alarm = ~true_defect & is_defect    # healthy sent down the cascade
    family_wrong = true_defect & is_defect & (fam != true_fam)
    fine_wrong = wrong & ~triage_miss & ~triage_false_alarm & ~family_wrong
    return {
        "stage1_missed_defects": int(triage_miss.sum()),
        "stage1_false_alarms": int(triage_false_alarm.sum()),
        "stage2_wrong_family": int(family_wrong.sum()),
        "stage3_wrong_subtype": int(fine_wrong.sum()),
        "total_errors": int(wrong.sum()),
        "early_exit_rate": float((~is_defect).mean()),
    }


@torch.inference_mode()
def latency_ms(fn, x: torch.Tensor, n: int = 300) -> float:
    """Mean batch-1 latency over `n` images, after a short warm-up."""
    idx = torch.randperm(len(x))[:n]
    for i in idx[:20]:
        fn(x[i:i + 1])
    t0 = time.perf_counter()
    for i in idx:
        fn(x[i:i + 1])
    return (time.perf_counter() - t0) / len(idx) * 1000


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--flat", action="store_true", help="also evaluate the flat baseline")
    args = ap.parse_args()

    cfg, tax = load_config(args.config), Taxonomy.load(args.taxonomy)
    run_dir, up = Path(cfg["train"]["out_dir"]), cfg["data"]["upsample"]
    splits, stats = load_splits(cfg, tax)
    data = splits[args.split]
    y = data.global_targets(tax)
    x = preprocess(to_batch(data.images), stats, up)  # CPU: matches edge-device conditions

    torch.set_num_threads(1)
    cascade = load_cascade(cfg, tax, run_dir)
    out = cascade.predict(x)
    results = {"split": args.split, "classes": tax.classes, "cascade": {
        **metrics(y, out["pred"].numpy(), tax.classes),
        "stages": stage_errors(y, out, tax),
        "params": n_params(cascade),
        "latency_ms_per_image": latency_ms(cascade.predict, x),
    }}

    if args.flat:
        flat = load_node(cfg, tax, "flat", run_dir)
        with torch.no_grad():
            pred = flat(x).argmax(1).numpy()
        results["flat"] = {**metrics(y, pred, tax.classes), "params": n_params(flat),
                           "latency_ms_per_image": latency_ms(flat, x)}

    for name in ("cascade", "flat"):
        if name in results:
            r = results[name]
            print(f"{name:8s} acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}  kappa {r['kappa']:.4f}  "
                  f"params {r['params']:,}  latency {r['latency_ms_per_image']:.2f} ms")
    print("cascade stage errors:", json.dumps(results["cascade"]["stages"]))

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {run_dir / 'results.json'}")


if __name__ == "__main__":
    main()
