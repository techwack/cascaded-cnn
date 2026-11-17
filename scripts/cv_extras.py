"""Computer-vision extras for the CV report (run with the project venv):
  1. classical hand-crafted features + SVM baseline -> results/handcrafted_svm.json
  2. first-layer filters and feature maps of CNN-1 -> docs/figures/cv_filters.png, cv_featuremaps.png
  3. Grad-CAM heat maps of Stage 1 and the specialists -> docs/figures/cv_gradcam.png
"""
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarnet.common import load_cascade, load_config, preprocess, to_batch  # noqa: E402
from solarnet.data import load_image, load_records  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402

OUT = ROOT / "docs" / "figures"
cfg, tax = load_config(ROOT / "configs" / "improved3.yaml"), Taxonomy.load(ROOT / "configs" / "taxonomy.yaml")
saved = json.load(open(cfg["data"]["splits_file"]))
recs = {r["id"]: r for r in load_records(cfg["data"]["root"])}
stats = saved["stats"]


# ------------------------------------------------------------------ 1. hand-crafted features
def components(mask):
    """4-connected component sizes of a boolean mask (simple BFS)."""
    seen = np.zeros_like(mask, bool)
    sizes = []
    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j] and not seen[i, j]:
                stack, n = [(i, j)], 0
                seen[i, j] = True
                while stack:
                    a, b = stack.pop()
                    n += 1
                    for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        c, d = a + da, b + db
                        if 0 <= c < h and 0 <= d < w and mask[c, d] and not seen[c, d]:
                            seen[c, d] = True
                            stack.append((c, d))
                sizes.append(n)
    return sizes


def features(x):
    f = [x.mean(), x.std(), x.min(), x.max()] + list(np.percentile(x, [5, 25, 50, 75, 95]))
    z = (x - x.mean()) / (x.std() + 1e-6)
    f += [(z ** 3).mean(), (z ** 4).mean()]
    h = x.shape[0] // 3
    f += [x[:h].mean(), x[h:2 * h].mean(), x[2 * h:].mean(), x[:, :12].mean(), x[:, 12:].mean(), x.mean(0).std(), x.mean(1).std()]
    gy, gx = np.gradient(x)
    mag, ang = np.hypot(gx, gy), np.arctan2(gy, gx)
    f += [mag.mean(), mag.std(), mag.max()]
    hist, _ = np.histogram(ang, bins=8, range=(-np.pi, np.pi), weights=mag)
    f += list(hist / (hist.sum() + 1e-6))
    for thr in (x.mean() + x.std(), x.mean() + 2 * x.std()):
        m = x > thr
        sz = components(m)
        ys, xs = np.nonzero(m) if m.any() else (np.array([20]), np.array([12]))
        f += [m.mean(), len(sz), max(sz) if sz else 0, ys.mean() / 40, xs.mean() / 24]
    f += [(x > 0.8).mean(), (x < 0.3).mean()]
    return np.array(f, dtype=np.float32)


def handcrafted():
    def feats(ids):
        return np.stack([features(load_image(recs[i]["path"])) for i in ids]), np.array([tax.index(recs[i]["label"]) for i in ids])
    t0 = time.time()
    Xtr, ytr = feats(saved["train"]); Xva, yva = feats(saved["val"]); Xte, yte = feats(saved["test"])
    sc = StandardScaler().fit(Xtr)
    best = (-1, None, None)
    for C in (1, 10, 100):
        clf = SVC(C=C, kernel="rbf", class_weight="balanced").fit(sc.transform(Xtr), ytr)
        f1 = f1_score(yva, clf.predict(sc.transform(Xva)), average="macro", zero_division=0)
        print(f"C={C} val macro-F1 {f1:.4f}", flush=True)
        if f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    pred = clf.predict(sc.transform(Xte))
    healthy = tax.index(tax.healthy)
    res = {"model": f"{Xtr.shape[1]} hand-crafted features + standardised RBF SVM", "n_features": int(Xtr.shape[1]), "C": C,
           "accuracy": accuracy_score(yte, pred), "macro_f1": f1_score(yte, pred, average="macro", zero_division=0), "kappa": cohen_kappa_score(yte, pred),
           "binary_accuracy": float(((pred != healthy) == (yte != healthy)).mean()), "seconds": round(time.time() - t0)}
    (ROOT / "results" / "handcrafted_svm.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ 2. filters and feature maps
def filters_and_maps(cascade, imgs):
    net = cascade.triage
    conv = [m for m in net.features if isinstance(m, torch.nn.Conv2d)][0]
    w = conv.weight.detach().cpu().numpy()[:, 0]  # 64 x 3 x 3
    fig, axes = plt.subplots(8, 8, figsize=(5.6, 5.6))
    for a, k in zip(axes.ravel(), w):
        a.imshow(k, cmap="coolwarm", vmin=-np.abs(w).max(), vmax=np.abs(w).max()); a.axis("off")
    fig.suptitle("First-layer 3 x 3 filters of CNN-1 (64 learned filters)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "cv_filters.png", dpi=200); plt.close(fig)

    x = preprocess(to_batch(imgs["Hot-Spot"][None]), stats, 2)
    with torch.no_grad():
        maps = {}
        h = x
        for i, layer in enumerate(net.features):
            h = layer(h)
            if i in (2, 6, 10):  # after the first ReLU of block 1, 2 and 3
                maps[i] = h[0].numpy()
    fig, axes = plt.subplots(3, 9, figsize=(9.5, 3.9))
    for r, (i, m) in enumerate(maps.items()):
        axes[r, 0].imshow(imgs["Hot-Spot"], cmap="inferno"); axes[r, 0].set_title("input" if r == 0 else "", fontsize=7)
        for c in range(1, 9):
            axes[r, c].imshow(m[c - 1], cmap="viridis")
        axes[r, 0].set_ylabel(f"block {r + 1}\n{m.shape[0]}x{m.shape[1]}x{m.shape[2]}", fontsize=7)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Feature maps of CNN-1 for a Hot-Spot image: the first 8 channels after the first ReLU of each block", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "cv_featuremaps.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------ 3. Grad-CAM
def gradcam(net, x, cls):
    feats = list(net.features)
    last_relu = max(i for i, m in enumerate(feats) if isinstance(m, torch.nn.ReLU))
    acts = {}

    def hook(_, __, out):
        out.retain_grad()
        acts["a"] = out
    h = feats[last_relu].register_forward_hook(hook)
    net.zero_grad()
    logits = net(x)
    score = logits[0, cls]
    score.backward()
    h.remove()
    a = acts["a"]
    wts = a.grad.mean((2, 3), keepdim=True)
    cam = F.relu((wts * a).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=(40, 24), mode="bilinear", align_corners=False)[0, 0].numpy()
    return cam / (cam.max() + 1e-8), logits.softmax(1)[0].detach().numpy()


def gradcam_figure(cascade, imgs):
    show = ["Hot-Spot", "Cell", "Diode", "Soiling", "Cracking", "Offline-Module"]
    fig, axes = plt.subplots(len(show), 4, figsize=(7.4, 1.9 * len(show)))
    for r, c in enumerate(show):
        img = imgs[c]
        x = preprocess(to_batch(img[None]), stats, 2)
        cam1, p1 = gradcam(cascade.triage, x, 1)
        fam = cascade.family_names[int(cascade.family(x).argmax())]
        net3 = cascade.fine[fam]
        out3 = net3(x)
        k = int(out3.argmax())
        cam3, p3 = gradcam(net3, x, k)
        pred = tax.families[fam][k]
        axes[r, 0].imshow(img, cmap="inferno"); axes[r, 0].set_title(f"true: {c}", fontsize=7)
        for col, (cam, ttl) in enumerate(((cam1, f"Stage 1: defect ({p1[1]:.0%})"), (cam3, f"Stage 3 ({fam}): {pred} ({p3[k]:.0%})")), 1):
            axes[r, col].imshow(img, cmap="gray"); axes[r, col].imshow(cam, cmap="jet", alpha=0.5); axes[r, col].set_title(ttl, fontsize=6.5)
        axes[r, 3].imshow(cam1 * cam3, cmap="jet"); axes[r, 3].set_title("overlap of the two maps", fontsize=6.5)
        for a in axes[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Grad-CAM: where the networks look (red = high influence on the decision)", fontsize=8.5)
    fig.tight_layout(); fig.savefig(OUT / "cv_gradcam.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    cascade = load_cascade(cfg, tax, ROOT / cfg["train"]["out_dir"]).eval()
    first = {}
    for i in saved["test"]:
        first.setdefault(recs[i]["label"], load_image(recs[i]["path"]))
    filters_and_maps(cascade, first)
    gradcam_figure(cascade, first)
    handcrafted()
    print(sorted(p.name for p in OUT.glob("cv_*.png")))
