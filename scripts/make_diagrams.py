"""Stage-level architecture diagrams (layer by layer) for the report.

    python scripts/make_diagrams.py        # writes docs/figures/*.png

Sizes follow configs/improved3.yaml (80x48 input after 2x upsampling, two 3x3 convs per block).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
GREY, WHITE, LIGHT = "#f2f2f2", "#ffffff", "#e6e6e6"
plt.rcParams["font.family"] = "DejaVu Sans"


def box(ax, cx, cy, w, h, text, fc=GREY, fs=8.5, bold=False, lw=1.3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="black", lw=lw))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=None, fs=7.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.22, label, ha="center", fontsize=fs, style="italic")


def cnn(name, subtitle, channels, dropout, head_label, out_lines, fname, in_shape="1 x 80 x 48"):
    """Horizontal layer-by-layer diagram of one StageCNN."""
    fig, ax = plt.subplots(figsize=(11.5, 4.3))
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.3); ax.axis("off")
    ax.text(5.75, 4.05, name, ha="center", fontsize=12, fontweight="bold")
    ax.text(5.75, 3.78, subtitle, ha="center", fontsize=8.5, style="italic")

    y, x = 2.2, 0.85
    box(ax, x, y, 1.45, 0.9, "INPUT\nIR image\n(x2 upsampled)", WHITE, 7.5, True)
    ax.text(x, y - 0.75, in_shape, ha="center", fontsize=7)
    h, w = 80, 48
    prev = x
    x += 1.65
    for i, ch in enumerate(channels, 1):
        arrow(ax, prev + (0.7 if i == 1 else 0.62), y, x - 0.62, y)
        box(ax, x, y, 1.2, 1.7, f"BLOCK {i}\n\nConv 3x3, {ch}\nBN + ReLU\nConv 3x3, {ch}\nBN + ReLU\nMaxPool 2x2", GREY, 7.3)
        h, w = h // 2, w // 2
        ax.text(x, y - 1.05, f"{ch} x {h} x {w}", ha="center", fontsize=7)
        prev, x = x, x + 1.6
    arrow(ax, prev + 0.62, y, x - 0.55, y)
    box(ax, x, y, 1.0, 0.9, "Global\nAvg Pool", LIGHT, 7.5)
    ax.text(x, y - 0.75, f"{channels[-1]}", ha="center", fontsize=7)
    prev, x = x, x + 1.3
    arrow(ax, prev + 0.5, y, x - 0.5, y)
    box(ax, x, y, 0.95, 0.9, f"Dropout\np = {dropout}", LIGHT, 7.5)
    prev, x = x, x + 1.3
    arrow(ax, prev + 0.48, y, x - 0.5, y)
    box(ax, x, y, 0.95, 0.9, head_label, LIGHT, 7.5)
    arrow(ax, x, y - 0.46, x, 0.95)
    box(ax, x, 0.55, 1.9, 0.75, "\n".join(out_lines), WHITE, 7.3, True)
    ax.text(0.15, 0.3, "Each block halves height and width. All convolutions use padding 1.",
            fontsize=7, style="italic")
    fig.savefig(OUT / fname, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def stage3(fname):
    """Stage 3: one shared design, three specialists differing only in the output layer."""
    fig, ax = plt.subplots(figsize=(12.5, 6.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(-0.4, 6.0); ax.axis("off")
    ax.text(6.25, 5.75, "STAGE 3  |  CNN-3a / 3b / 3c  (fine-grained specialists)", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.25, 5.48, "One specialist per family. Same backbone, different output layer.", ha="center", fontsize=8.5, style="italic")
    box(ax, 6.25, 4.8, 4.4, 0.55, "Family chosen by Stage 2 routes the image", WHITE, 8, True)
    xs = [2.15, 6.25, 10.35]
    specs = [("CNN-3a  ELECTRICAL", "FC (6)", ["Cell", "Cell-Multi", "Diode", "Diode-Multi", "Hot-Spot", "Hot-Spot-Multi"]),
             ("CNN-3b  ENVIRONMENTAL", "FC (3)", ["Soiling", "Shadowing", "Vegetation"]),
             ("CNN-3c  STRUCTURAL", "FC (2)", ["Cracking", "Offline-Module"])]
    body = ("Block 1: 2 x Conv3x3(128) + MaxPool -> 128 x 40 x 24\n"
            "Block 2: 2 x Conv3x3(256) + MaxPool -> 256 x 20 x 12\n"
            "Block 3: 2 x Conv3x3(128) + MaxPool -> 128 x 10 x 6")
    for x, (title, fc, cls) in zip(xs, specs):
        arrow(ax, 6.25 + (x - 6.25) * 0.25, 4.52, x, 3.98)
        box(ax, x, 3.45, 3.9, 1.05, title + "\n" + body, GREY, 6.6)
        arrow(ax, x, 2.92, x, 2.62)
        box(ax, x, 2.35, 3.9, 0.5, "Global Avg Pool (128) > Dropout 0.4 > " + fc, LIGHT, 7.2)
        arrow(ax, x, 2.1, x, 1.78)
        h = 0.3 * len(cls) + 0.3
        box(ax, x, 1.78 - h / 2 - 0.02, 2.4, h, "\n".join(cls), WHITE, 7.6, True)
    ax.text(6.25, -0.3, "Output: final defect class (global label = family offset + specialist prediction)", ha="center", fontsize=8, style="italic")
    fig.savefig(OUT / fname, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cnn("STAGE 1  |  CNN-1  (binary triage)", "Decides No-Anomaly vs Defect; No-Anomaly exits early",
        [64, 128, 128], 0.2, "FC (2)", ["No-Anomaly -> STOP", "Defect -> Stage 2"], "stage1_cnn1.png")
    cnn("STAGE 2  |  CNN-2  (fault-family classifier)", "Routes each defect to one of three families",
        [128, 256, 128], 0.4, "FC (3)", ["Electrical", "Environmental", "Structural"], "stage2_cnn2.png")
    stage3("stage3_specialists.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
