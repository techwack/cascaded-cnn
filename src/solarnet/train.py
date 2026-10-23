"""Train one node of the cascade (or the flat baseline).

    python -m solarnet.train --node triage
    python -m solarnet.train --node family
    python -m solarnet.train --node electrical      # one per family in taxonomy.yaml
    python -m solarnet.train --node flat            # single-stage 12-class baseline
    python -m solarnet.train --node all             # every cascade node in turn

Each node is trained on ground-truth routing: the family CNN sees every
defect image, the Electrical CNN sees only Electrical images, and so on.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .common import augment, build_model, device, load_config, preprocess, save_checkpoint, seed_everything, to_batch
from .data import class_weights, load_splits
from .taxonomy import Taxonomy


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, stats: dict, upsample: int, dev: torch.device,
            batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = [model(preprocess(x[i:i + batch_size].to(dev), stats, upsample)).argmax(1).cpu()
           for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


def train_node(cfg: dict, tax: Taxonomy, node: str, splits: dict, stats: dict) -> dict:
    seed_everything(cfg["data"]["seed"])
    dev, tc, up = device(), cfg["train"], cfg["data"]["upsample"]
    classes = tax.node_classes(node)

    x_tr, y_tr = splits["train"].for_node(tax, node)
    x_va, y_va = splits["val"].for_node(tax, node)
    print(f"[{node}] classes={classes}")
    print(f"[{node}] train={len(y_tr)} val={len(y_va)} counts={np.bincount(y_tr, minlength=len(classes)).tolist()}")

    sampler = None
    if tc.get("sampler", "none") == "sqrt":  # rare classes drawn more often: weight ~ 1/sqrt(class count)
        counts = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
        sample_w = torch.from_numpy((1.0 / np.sqrt(np.maximum(counts, 1)))[y_tr])
        sampler = WeightedRandomSampler(sample_w, num_samples=len(y_tr), replacement=True)
    loader = DataLoader(TensorDataset(to_batch(x_tr), torch.from_numpy(y_tr)),
                        batch_size=tc["batch_size"], shuffle=sampler is None, sampler=sampler, drop_last=False)
    x_va_t = to_batch(x_va)

    model = build_model(cfg, tax, node).to(dev)
    weight = torch.from_numpy(class_weights(y_tr, len(classes))).to(dev) if tc["weighted_loss"] else None
    loss_fn = nn.CrossEntropyLoss(weight=weight, label_smoothing=tc.get("label_smoothing", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=tc["patience"] // 3)

    out_path = Path(tc["out_dir"]) / node / "best.pt"
    best, best_epoch, history = -1.0, 0, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            xb = preprocess(augment(xb, cfg["augment"]), stats, up)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
            n += len(yb)

        pred = predict(model, x_va_t, stats, up, dev)
        f1 = f1_score(y_va, pred, average="macro", zero_division=0)
        acc = float((pred == y_va).mean())
        sched.step(f1)
        history.append({"epoch": epoch, "loss": total / n, "val_acc": acc, "val_macro_f1": f1})
        print(f"[{node}] ep {epoch:3d} loss {total / n:.4f} val_acc {acc:.4f} val_f1 {f1:.4f} ({time.time() - t0:.1f}s)")

        if f1 > best:
            best, best_epoch = f1, epoch
            save_checkpoint(out_path, model, node, classes, stats, up,
                            {"val_macro_f1": f1, "val_acc": acc, "epoch": epoch})
        elif epoch - best_epoch >= tc["patience"]:
            print(f"[{node}] early stop at epoch {epoch}; best epoch {best_epoch} (val_f1 {best:.4f})")
            break

    with open(out_path.parent / "history.json", "w") as f:
        json.dump(history, f, indent=1)
    return {"node": node, "best_val_macro_f1": best, "best_epoch": best_epoch,
            "params": sum(p.numel() for p in model.parameters())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", required=True, help="triage | family | <family> | flat | all")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--taxonomy", default="configs/taxonomy.yaml")
    args = ap.parse_args()

    cfg, tax = load_config(args.config), Taxonomy.load(args.taxonomy)
    nodes = tax.nodes if args.node == "all" else [args.node]
    for node in nodes:  # fail fast on a typo before loading any images
        tax.node_classes(node)

    splits, stats = load_splits(cfg, tax)
    summary = [train_node(cfg, tax, node, splits, stats) for node in nodes]
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
