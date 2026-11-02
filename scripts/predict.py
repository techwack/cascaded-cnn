"""Live demo: watch one infrared image go through the cascade, stage by stage.

    python scripts/predict.py --sample 4                    # 4 random test images
    python scripts/predict.py --class Hot-Spot --sample 3   # 3 random Hot-Spot test images
    python scripts/predict.py --image path/to/module.jpg    # any image (resized to 40x24 grayscale)

Needs trained checkpoints in the run directory of the config (default configs/improved3.yaml ->
runs_improved3/) and data/splits.json (for the normalisation statistics). Writes a figure to
demo_out/ and opens it unless --no-show is given.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarnet.common import load_cascade, load_config, preprocess, to_batch  # noqa: E402
from solarnet.data import IMG_H, IMG_W, load_image, load_records  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402

DARK, LIGHT, RED = "#222222", "#bbbbbb", "#c0392b"


def read_any(path: str) -> np.ndarray:
    """Grayscale [0,1] array of shape (40, 24); other sizes are resized."""
    try:
        return load_image(path)
    except ValueError:
        img = Image.open(path).convert("L").resize((IMG_W, IMG_H))
        return np.asarray(img, dtype=np.float32) / 255.0


@torch.no_grad()
def explain(cascade, x: torch.Tensor, tax: Taxonomy) -> dict:
    """Run one preprocessed image [1,1,H,W] through the cascade and keep every stage's probabilities."""
    probs = cascade._probs
    p1 = probs(cascade.triage, x, False)[0]
    out = {"p_stage1": p1.tolist(), "exited": bool(p1[1] <= 0.5), "p_stage2": None, "family": None,
           "p_stage3": None, "classes3": None}
    if out["exited"]:
        out["pred"] = tax.healthy
        return out
    p2 = probs(cascade.family, x, False)[0]
    fam = cascade.family_names[int(p2.argmax())]
    p3 = probs(cascade.fine[fam], x, False)[0]
    out.update(p_stage2=p2.tolist(), family=fam, p_stage3=p3.tolist(), classes3=tax.families[fam],
               pred=tax.families[fam][int(p3.argmax())])
    return out


def bars(ax, labels, values, chosen, title):
    colors = [DARK if i == chosen else LIGHT for i in range(len(labels))]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=9, fontweight="bold")
    for i, v in enumerate(values):
        ax.text(min(v + 0.02, 0.78), i, f"{v:.0%}", va="center", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def skipped(ax, title, why):
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", color="#888888")
    ax.text(0.5, 0.5, why, ha="center", va="center", fontsize=9, color="#888888", style="italic",
            transform=ax.transAxes)


def draw(rows, tax, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(11.5, 2.5 * n + 0.6), squeeze=False,
                             gridspec_kw={"width_ratios": [0.9, 1, 1.15, 1.5]})
    for r, (name, img, truth, res) in enumerate(rows):
        a0, a1, a2, a3 = axes[r]
        a0.imshow(img, cmap="inferno", interpolation="nearest")
        a0.set_xticks([]); a0.set_yticks([])
        ok = truth is None or truth == res["pred"]
        head = f"Prediction: {res['pred']}" + ("" if truth is None else f"\nTrue: {truth}")
        a0.set_title(head, fontsize=9, fontweight="bold", color="#1e8449" if ok else RED)
        bars(a1, tax.classes[:1] + ["Defect"], res["p_stage1"], int(res["p_stage1"][1] > 0.5), "Stage 1: triage")
        if res["exited"]:
            skipped(a2, "Stage 2: family", "skipped\n(early exit)")
            skipped(a3, "Stage 3: specialist", "skipped\n(early exit)")
        else:
            fams = list(tax.families)
            bars(a2, fams, res["p_stage2"], fams.index(res["family"]), "Stage 2: family")
            bars(a3, res["classes3"], res["p_stage3"], res["classes3"].index(res["pred"]),
                 f"Stage 3: {res['family']} specialist")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/improved3.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    ap.add_argument("--image", nargs="*", default=[], help="path(s) to image file(s)")
    ap.add_argument("--sample", type=int, default=0, help="number of random test images")
    ap.add_argument("--class", dest="cls", default=None, help="restrict --sample to this true class")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()
    if not args.image and not args.sample:
        args.sample = 4
    matplotlib.use("Agg")

    cfg, tax = load_config(args.config), Taxonomy.load(args.taxonomy)
    run_dir, up = Path(cfg["train"]["out_dir"]), cfg["data"]["upsample"]
    saved = json.load(open(cfg["data"]["splits_file"]))
    stats = saved["stats"]
    cascade = load_cascade(cfg, tax, run_dir)

    items = [(Path(p).name, read_any(p), None) for p in args.image]
    if args.sample:
        records = {r["id"]: r for r in load_records(cfg["data"]["root"])}
        pool = [records[i] for i in saved["test"] if args.cls is None or records[i]["label"] == args.cls]
        if not pool:
            sys.exit(f"no test images for class {args.cls!r}; classes: {tax.classes}")
        rng = np.random.default_rng(args.seed)
        for r in rng.choice(len(pool), size=min(args.sample, len(pool)), replace=False):
            rec = pool[int(r)]
            items.append((f"test #{rec['id']}", read_any(rec["path"]), rec["label"]))

    rows = []
    for name, img, truth in items:
        x = preprocess(to_batch(img[None]), stats, up)
        res = explain(cascade, x, tax)
        rows.append((name, img, truth, res))
        s1 = res["p_stage1"][1]
        line = f"{name:14s} stage1 P(defect)={s1:.2f}"
        line += "  -> No-Anomaly (early exit)" if res["exited"] else \
            f"  stage2 -> {res['family']} ({max(res['p_stage2']):.2f})  stage3 -> {res['pred']} ({max(res['p_stage3']):.2f})"
        if truth is not None:
            line += f"   [true: {truth}] " + ("OK" if truth == res["pred"] else "WRONG")
        print(line)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / ("predict_" + (args.cls or "mix").lower() + ".png")
    draw(rows, tax, out_path)
    print("saved", out_path)
    if not args.no_show and hasattr(os, "startfile"):
        os.startfile(out_path)


if __name__ == "__main__":
    main()
