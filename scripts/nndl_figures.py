"""Schematic figures for the NNDL case study report (theory chapter): receptive field, transfer learning, SE attention, GAN."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
GREY, WHITE, LIGHT, DARK = "#f2f2f2", "#ffffff", "#e0e0e0", "#2b3a4a"


def box(ax, cx, cy, w, h, text, fc=GREY, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="black", lw=1.2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=7.5, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def receptive_field():
    layers = ["Conv", "Conv", "Pool", "Conv", "Conv", "Pool", "Conv", "Conv", "Pool"]
    rf = [3, 5, 6, 10, 14, 16, 24, 32, 36]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.step(range(1, 10), rf, where="mid", color=DARK, lw=2)
    ax.scatter(range(1, 10), rf, color="#d9480f", zorder=3)
    for i, v in enumerate(rf, 1):
        ax.text(i, v + 1.2, str(v), ha="center", fontsize=8)
    ax.set_xticks(range(1, 10)); ax.set_xticklabels([f"{l}\\n(B{(i - 1) // 3 + 1})".replace("\\n", "\n") for i, l in enumerate(layers, 1)], fontsize=7)
    ax.set_ylabel("Receptive field (pixels of the 80 x 48 input)", fontsize=8); ax.set_ylim(0, 42)
    ax.set_title("Growth of the receptive field through the three blocks of CNN-1", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "nndl_receptive_field.png", dpi=200); plt.close(fig)


def transfer_learning():
    fig, ax = canvas(10.5, 4.0)
    box(ax, 1.6, 3.0, 2.8, 0.9, "Source task\nImageNet: 1.28 M natural\nRGB photographs, 1000 classes", WHITE, 8)
    box(ax, 5.2, 3.0, 3.2, 0.9, "Pretrained EfficientNet-B0\ngeneric filters (edges, blobs,\ntextures) + source classifier", GREY, 8)
    box(ax, 9.0, 3.0, 2.6, 0.9, "Target task\n24 x 40 thermal images,\n2 to 6 classes per stage", WHITE, 8)
    arrow(ax, 3.0, 3.0, 3.6, 3.0, "train"); arrow(ax, 6.8, 3.0, 7.7, 3.0, "adapt")
    box(ax, 3.3, 1.2, 2.8, 0.9, "Keep: convolutional\nfeature extractor", LIGHT, 8)
    box(ax, 6.9, 1.2, 2.8, 0.9, "Replace: classifier head\n(new FC layer, new dropout)", LIGHT, 8)
    box(ax, 9.6, 1.2, 1.7, 0.9, "Fine-tune all\nlayers\nlr = 3e-4", GREY, 8)
    arrow(ax, 5.2, 2.5, 3.6, 1.65); arrow(ax, 5.2, 2.5, 6.6, 1.65); arrow(ax, 8.3, 1.2, 8.75, 1.2)
    fig.suptitle("Transfer learning from ImageNet to thermal defect classification", fontsize=9, y=0.98)
    fig.savefig(OUT / "nndl_transfer_learning.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


def se_block():
    fig, ax = canvas(10.5, 3.2)
    box(ax, 1.0, 1.6, 1.5, 1.2, "Feature map X\nH x W x C", WHITE, 8)
    box(ax, 3.2, 1.6, 1.6, 1.2, "Squeeze\nglobal average\npool: z (C)", GREY, 8)
    box(ax, 5.3, 1.6, 1.8, 1.2, "Excitation\nFC (C/r) + ReLU\nFC (C) + sigmoid", GREY, 8)
    box(ax, 7.5, 1.6, 1.6, 1.2, "Channel\nweights s (C)", LIGHT, 8)
    box(ax, 9.6, 1.6, 1.6, 1.2, "Re-scaled map\nX' = s * X", WHITE, 8)
    for x1, x2 in ((1.75, 2.4), (4.0, 4.4), (6.2, 6.7), (8.3, 8.8)):
        arrow(ax, x1, 1.6, x2, 1.6)
    ax.annotate("", xy=(9.6, 0.95), xytext=(1.0, 0.95), arrowprops=dict(arrowstyle="-|>", lw=1.0, ls="--", color="gray", connectionstyle="arc3,rad=0"))
    ax.text(5.3, 0.62, "identity path: X is multiplied channel-wise by s", ha="center", fontsize=7.5, style="italic")
    fig.suptitle("Squeeze-and-excitation: channel attention inside the MBConv blocks of EfficientNet-B0", fontsize=9, y=0.98)
    fig.savefig(OUT / "nndl_se_attention.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


def gan():
    fig, ax = canvas(10.5, 3.8)
    box(ax, 1.2, 2.9, 1.8, 0.8, "Noise z\n~ N(0, I)", WHITE, 8)
    box(ax, 3.8, 2.9, 2.0, 0.9, "Generator G\nz -> fake image", GREY, 8)
    box(ax, 6.4, 1.9, 2.2, 1.0, "Discriminator D\nreal or fake?", GREY, 8)
    box(ax, 3.8, 0.8, 2.0, 0.8, "Real images\n(rare class)", WHITE, 8)
    box(ax, 9.3, 1.9, 1.8, 0.9, "Loss\n(adversarial)", LIGHT, 8)
    arrow(ax, 2.1, 2.9, 2.8, 2.9); arrow(ax, 4.8, 2.9, 5.5, 2.3, "fake"); arrow(ax, 4.8, 0.9, 5.5, 1.5, "real"); arrow(ax, 7.5, 1.9, 8.4, 1.9)
    ax.annotate("", xy=(3.8, 2.4), xytext=(9.3, 2.4), arrowprops=dict(arrowstyle="-|>", lw=1.0, ls="--", color="gray", connectionstyle="arc3,rad=-0.25"))
    ax.text(6.7, 3.3, "gradient to update G", ha="center", fontsize=7.5, style="italic")
    fig.suptitle("Generative adversarial network: G and D play a minimax game", fontsize=9, y=0.98)
    fig.savefig(OUT / "nndl_gan.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    receptive_field(); transfer_learning(); se_block(); gan()
    print(sorted(p.name for p in OUT.glob("nndl_*.png")))
