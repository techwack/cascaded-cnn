"""Duranay-style baseline on OUR split: ImageNet-pretrained EfficientNet-B0 features + SVM.

    python scripts/pretrained_baseline.py

Extracts 1280-d global-pooled features for every image (resized to 224x224, replicated to 3 channels),
fits a standardised RBF SVM on the train split, picks C on the validation split, and reports test
accuracy / macro-F1 / kappa plus healthy-vs-defect binary accuracy. Writes results/pretrained_svm.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from solarnet.common import device, load_config  # noqa: E402
from solarnet.data import load_image, load_records  # noqa: E402
from solarnet.taxonomy import Taxonomy  # noqa: E402


@torch.no_grad()
def extract(model, imgs: np.ndarray, dev, bs: int = 256) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(dev).unsqueeze(1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            f = model((x - mean) / std)
        out.append(f.float().cpu())
    return torch.cat(out).numpy()


def main() -> None:
    cfg, tax = load_config("configs/improved3.yaml"), Taxonomy.load("configs/taxonomy.yaml")
    saved = json.load(open(cfg["data"]["splits_file"]))
    rec = {r["id"]: r for r in load_records(cfg["data"]["root"])}
    dev = device()
    net = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    net.classifier = torch.nn.Identity()
    net = net.to(dev).eval()

    data = {}
    t0 = time.time()
    for sp in ("train", "val", "test"):
        ids = saved[sp]
        imgs = np.stack([load_image(rec[i]["path"]) for i in ids])
        data[sp] = (extract(net, imgs, dev), np.array([tax.index(rec[i]["label"]) for i in ids]))
        print(f"{sp}: features {data[sp][0].shape} ({time.time() - t0:.0f}s)", flush=True)

    sc = StandardScaler().fit(data["train"][0])
    X = {k: sc.transform(v[0]) for k, v in data.items()}
    y = {k: v[1] for k, v in data.items()}
    best = (-1, None, None)
    for C in (1, 10, 100):
        clf = SVC(C=C, kernel="rbf", class_weight="balanced").fit(X["train"], y["train"])
        f1 = f1_score(y["val"], clf.predict(X["val"]), average="macro", zero_division=0)
        print(f"C={C}: val macro-F1 {f1:.4f}", flush=True)
        if f1 > best[0]:
            best = (f1, C, clf)
    _, C, clf = best
    pred = clf.predict(X["test"])
    healthy = tax.index(tax.healthy)
    res = {"model": "EfficientNet-B0 (ImageNet) features + standardised RBF SVM", "C": C,
           "accuracy": accuracy_score(y["test"], pred), "macro_f1": f1_score(y["test"], pred, average="macro", zero_division=0),
           "kappa": cohen_kappa_score(y["test"], pred),
           "binary_accuracy": float(((pred != healthy) == (y["test"] != healthy)).mean())}
    print(json.dumps(res, indent=1))
    Path("results/pretrained_svm.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
