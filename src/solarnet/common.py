"""Config, model construction, checkpoints, and tensor transforms shared by train/eval."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .models import CascadedSolarNet, StageCNN
from .taxonomy import Taxonomy, node_name


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_kind(node: str) -> str:
    """Which entry of `models:` in the config a node uses."""
    return node if node in ("triage", "family", "flat") else "fine"


def build_model(cfg: dict, tax: Taxonomy, node: str) -> StageCNN:
    spec = cfg["models"][model_kind(node)]
    return StageCNN(spec["channels"], len(tax.node_classes(node)), spec["dropout"])


def save_checkpoint(path: Path, model: torch.nn.Module, node: str, classes: list[str], stats: dict,
                    upsample: int, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "node": node, "classes": classes,
                "stats": stats, "upsample": upsample, "metrics": metrics}, path)


def load_node(cfg: dict, tax: Taxonomy, node: str, run_dir: Path) -> StageCNN:
    ckpt = torch.load(run_dir / node / "best.pt", map_location="cpu")
    if ckpt["classes"] != tax.node_classes(node):
        raise ValueError(f"{node}: checkpoint classes {ckpt['classes']} != taxonomy {tax.node_classes(node)}")
    model = build_model(cfg, tax, node)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def load_cascade(cfg: dict, tax: Taxonomy, run_dir: Path) -> CascadedSolarNet:
    offsets, start = {}, 1  # global index 0 is the healthy class
    for fam, cs in tax.families.items():
        offsets[fam] = start
        start += len(cs)
    fine = {fam: load_node(cfg, tax, node_name(fam), run_dir) for fam in tax.families}
    return CascadedSolarNet(
        triage=load_node(cfg, tax, "triage", run_dir),
        family=load_node(cfg, tax, "family", run_dir),
        fine=fine,
        family_names=tax.family_names,
        class_offsets=offsets,
        healthy_index=tax.index(tax.healthy),
    ).eval()


# ---- transforms -----------------------------------------------------------

def augment(x: torch.Tensor, aug: dict) -> torch.Tensor:
    """Stochastic train-time augmentation on a [B,1,H,W] batch in [0,1].

    90-degree rotations from the data report are left out: the images are
    40x24, so a quarter turn would change the shape. Two flips together
    already give the 180-degree rotation.
    """
    b = x.shape[0]
    flip_h = torch.rand(b, device=x.device) < aug["hflip"]
    x = torch.where(flip_h[:, None, None, None], x.flip(-1), x)
    flip_v = torch.rand(b, device=x.device) < aug["vflip"]
    x = torch.where(flip_v[:, None, None, None], x.flip(-2), x)

    def jitter(scale: float) -> torch.Tensor:
        return 1 + (torch.rand(b, 1, 1, 1, device=x.device) * 2 - 1) * scale

    x = x * jitter(aug["brightness"])
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * jitter(aug["contrast"]) + mean
    x = x + torch.randn_like(x) * aug["noise_std"]
    return x.clamp(0, 1)


def preprocess(x: torch.Tensor, stats: dict, upsample: int) -> torch.Tensor:
    """z-score with training-set stats, then optional bilinear upsampling."""
    x = (x - stats["mean"]) / stats["std"]
    if upsample > 1:
        x = F.interpolate(x, scale_factor=upsample, mode="bilinear", align_corners=False)
    return x


def to_batch(images: np.ndarray) -> torch.Tensor:
    """[N,H,W] float array in [0,1] -> [N,1,H,W] tensor."""
    return torch.from_numpy(images).unsqueeze(1)
