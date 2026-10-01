"""InfraredSolarModules loading, stratified splits, normalisation and augmentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .taxonomy import Taxonomy

IMG_H, IMG_W = 40, 24  # images are 24 px wide x 40 px tall


def load_records(root: str | Path) -> list[dict]:
    """Read module_metadata.json -> [{id, path, label}] sorted by id."""
    root = Path(root)
    with open(root / "module_metadata.json") as f:
        meta = json.load(f)
    return [
        {"id": str(k), "path": str(root / v["image_filepath"]), "label": v["anomaly_class"]}
        for k, v in sorted(meta.items(), key=lambda kv: int(kv[0]))
    ]


def make_splits(records: list[dict], val_frac: float, test_frac: float, seed: int) -> dict[str, list[str]]:
    """Stratified train/val/test split of record ids."""
    ids = [r["id"] for r in records]
    labels = [r["label"] for r in records]
    train_ids, rest_ids, _, rest_y = train_test_split(
        ids, labels, test_size=val_frac + test_frac, stratify=labels, random_state=seed
    )
    val_ids, test_ids = train_test_split(
        rest_ids, test_size=test_frac / (val_frac + test_frac), stratify=rest_y, random_state=seed
    )
    return {"train": sorted(train_ids, key=int), "val": sorted(val_ids, key=int), "test": sorted(test_ids, key=int)}


def load_image(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if img.shape != (IMG_H, IMG_W):
        raise ValueError(f"{path}: expected {(IMG_H, IMG_W)}, got {img.shape}")
    return img


class SplitData:
    """All images of one split held in memory (20k x 960 px is ~77 MB as float32)."""

    def __init__(self, records: list[dict], ids: list[str]):
        by_id = {r["id"]: r for r in records}
        self.records = [by_id[i] for i in ids]
        self.images = np.stack([load_image(r["path"]) for r in self.records])
        self.labels = [r["label"] for r in self.records]

    def for_node(self, tax: Taxonomy, node: str) -> tuple[np.ndarray, np.ndarray]:
        """Images and targets this node is trained on (ground-truth routing)."""
        targets = [tax.node_label(node, lab) for lab in self.labels]
        keep = np.array([t is not None for t in targets])
        return self.images[keep], np.array([t for t in targets if t is not None], dtype=np.int64)

    def global_targets(self, tax: Taxonomy) -> np.ndarray:
        return np.array([tax.index(lab) for lab in self.labels], dtype=np.int64)


def load_splits(cfg: dict, tax: Taxonomy) -> tuple[dict[str, SplitData], dict]:
    """Load the saved split file and the images for each split."""
    with open(cfg["data"]["splits_file"]) as f:
        saved = json.load(f)
    records = load_records(cfg["data"]["root"])
    unknown = {r["label"] for r in records} - set(tax.classes)
    if unknown:
        raise ValueError(f"labels in the dataset but not in taxonomy.yaml: {sorted(unknown)}")
    splits = {name: SplitData(records, saved[name]) for name in ("train", "val", "test")}
    return splits, saved["stats"]


def class_weights(targets: np.ndarray, num_classes: int) -> np.ndarray:
    """w_r = N / (K * n_r); classes absent from `targets` get weight 0."""
    counts = np.bincount(targets, minlength=num_classes).astype(np.float64)
    with np.errstate(divide="ignore"):
        w = np.where(counts > 0, len(targets) / (num_classes * counts), 0.0)
    return w.astype(np.float32)
