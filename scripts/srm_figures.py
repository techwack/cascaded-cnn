"""Extra figures for the final report: class distribution, sample images, confusion matrices, training curves.

    python scripts/srm_figures.py        # writes docs/figures/*.png (needs the dataset and runs_* folders for some figures)
"""
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = {k: json.load(open(ROOT / "results" / f"{k}.json")) for k in ("pretrained", "improved3")}
CLASSES = R["pretrained"]["classes"]
INK, LIGHT, ACC = "#2b3a4a", "#adb5bd", "#d9480f"


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def class_distribution():
    totals = [10000, 1877, 1288, 1499, 175, 249, 246, 204, 1056, 1639, 940, 827]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.bar(range(12), totals, color=[LIGHT] + [INK] * 11)
    ax.set_yscale("log")
    for i, v in enumerate(totals):
        ax.text(i, v * 1.1, str(v), ha="center", fontsize=7)
    ax.set_xticks(range(12)); ax.set_xticklabels(CLASSES, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Images (log scale)", fontsize=8); ax.set_title("Class distribution of InfraredSolarModules", fontsize=9); clean(ax)
    fig.tight_layout(); fig.savefig(OUT / "class_distribution.png", dpi=200); plt.close(fig)


def sample_images():
    root = ROOT / "data" / "InfraredSolarModules"
    meta = json.load(open(root / "module_metadata.json"))
    test = json.load(open(ROOT / "data" / "splits.json"))["test"]
    first = {}
    for i in test:
        lab = meta[i]["anomaly_class"]
        first.setdefault(lab, i)
    fig, axes = plt.subplots(2, 6, figsize=(9, 4.4))
    for ax, c in zip(axes.ravel(), CLASSES):
        img = np.asarray(Image.open(root / meta[first[c]]["image_filepath"]).convert("L"), dtype=float) / 255
        ax.imshow(img, cmap="inferno", interpolation="nearest"); ax.set_title(c, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("One test image per class (40 x 24 pixels, inferno colour map)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "sample_images.png", dpi=200); plt.close(fig)


def confusion(res, name, title):
    cm = np.array(res["confusion"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(12):
        for j in range(12):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6, color="white" if norm[i, j] > 0.55 else "black")
    ax.set_xticks(range(12)); ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(12)); ax.set_yticklabels(CLASSES, fontsize=7)
    ax.set_xlabel("Predicted class", fontsize=8); ax.set_ylabel("True class", fontsize=8); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("Row-normalised", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig)


def curves():
    nodes = ["triage", "family", "electrical", "environmental", "structural"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6))
    best = {}
    for col, (label, d) in enumerate((("From scratch (improved3)", "runs_improved3"), ("Pretrained EfficientNet-B0", "runs_pretrained"))):
        for node in nodes:
            f = ROOT / d / node / "history.json"
            if not f.exists():
                continue
            h = json.load(open(f))
            ep = [e["epoch"] for e in h]
            axes[0, col].plot(ep, [e["val_macro_f1"] for e in h], label=node, lw=1.2)
            axes[1, col].plot(ep, [e["loss"] for e in h], label=node, lw=1.2)
            best[(label, node)] = max(e["val_macro_f1"] for e in h)
        axes[0, col].set_title(label + ": validation macro-F1", fontsize=9)
        axes[1, col].set_title(label + ": training loss", fontsize=9)
        for r in (0, 1):
            axes[r, col].set_xlabel("Epoch", fontsize=8); axes[r, col].tick_params(labelsize=7); clean(axes[r, col])
        axes[0, col].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=200); plt.close(fig)
    # best validation F1 per node
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(nodes)); w = 0.38
    s = [best.get(("From scratch (improved3)", n), 0) for n in nodes]
    p = [best.get(("Pretrained EfficientNet-B0", n), 0) for n in nodes]
    b1 = ax.bar(x - w / 2, s, w, color=LIGHT, label="From scratch"); b2 = ax.bar(x + w / 2, p, w, color=INK, label="Pretrained")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(nodes, fontsize=8); ax.set_ylim(0, 1.1); ax.set_ylabel("Best validation macro-F1", fontsize=8)
    ax.legend(fontsize=7); ax.set_title("Per-node validation macro-F1", fontsize=9); clean(ax)
    fig.tight_layout(); fig.savefig(OUT / "node_comparison.png", dpi=200); plt.close(fig)
    json.dump({f"{a}|{b}": v for (a, b), v in best.items()}, open(ROOT / "results" / "node_val_f1.json", "w"), indent=1)


if __name__ == "__main__":
    class_distribution()
    try:
        sample_images()
    except Exception as e:  # dataset not present
        print("sample images skipped:", e)
    confusion(R["pretrained"]["cascade"], "confusion_cascade.png", "Pretrained cascade: confusion matrix (test set)")
    confusion(R["pretrained"]["flat"], "confusion_flat.png", "Pretrained flat CNN: confusion matrix (test set)")
    curves()
    print(sorted(p.name for p in OUT.glob("*.png")))
