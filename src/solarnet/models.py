"""Stage CNNs and the full 3-stage cascade."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageCNN(nn.Module):
    """[Conv3x3-BN-ReLU-MaxPool2] x len(channels) -> GlobalAvgPool -> Dropout -> FC.

    With the native 40x24 input, three blocks leave a 5x3 map before pooling.
    """

    def __init__(self, channels: list[int], num_classes: int, dropout: float = 0.0, in_ch: int = 1,
                 convs_per_block: int = 1):
        super().__init__()
        layers, prev = [], in_ch
        for ch in channels:
            for _ in range(convs_per_block):
                layers += [
                    nn.Conv2d(prev, ch, 3, padding=1, bias=False),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(inplace=True),
                ]
                prev = ch
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(prev, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.head(F.adaptive_avg_pool2d(x, 1))


class CascadedSolarNet(nn.Module):
    """CNN-1 (triage) -> CNN-2 (family) -> CNN-3x (fine-grained, one per family).

    `fine` maps family name -> model, in the same order as `family_names`.
    `class_offsets[f]` is the global class index of family f's first class,
    so a fine prediction p in family f is global class offset + p.
    """

    def __init__(self, triage: nn.Module, family: nn.Module, fine: dict[str, nn.Module],
                 family_names: list[str], class_offsets: dict[str, int], healthy_index: int = 0):
        super().__init__()
        self.triage, self.family = triage, family
        self.fine = nn.ModuleDict(fine)
        self.family_names = family_names
        self.class_offsets = class_offsets
        self.healthy_index = healthy_index

    @staticmethod
    def _probs(model: nn.Module, x: torch.Tensor, tta: bool) -> torch.Tensor:
        """Softmax probabilities, optionally averaged over flip views (test-time augmentation)."""
        views = [x] + ([x.flip(-1), x.flip(-2), x.flip(-1).flip(-2)] if tta else [])
        return torch.stack([F.softmax(model(v), dim=1) for v in views]).mean(0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5, tta: bool = False) -> dict[str, torch.Tensor]:
        """Route a batch through the cascade (hard routing, with early exit).

        `threshold` is the triage P(defect) above which an image continues to stage 2.
        Returns global class predictions plus the per-stage decisions, so
        evaluation can attribute errors to the stage that made them.
        """
        n = x.shape[0]
        pred = torch.full((n,), self.healthy_index, dtype=torch.long, device=x.device)
        fam_pred = torch.full((n,), -1, dtype=torch.long, device=x.device)

        is_defect = self._probs(self.triage, x, tta)[:, 1] > threshold
        idx = is_defect.nonzero(as_tuple=True)[0]
        if idx.numel():  # early exit: healthy panels never reach stage 2
            fam_pred[idx] = self._probs(self.family, x[idx], tta).argmax(1)
            for f_i, fam in enumerate(self.family_names):
                sub = idx[fam_pred[idx] == f_i]
                if sub.numel():
                    pred[sub] = self.class_offsets[fam] + self._probs(self.fine[fam], x[sub], tta).argmax(1)
        return {"pred": pred, "is_defect": is_defect, "family": fam_pred}

    @torch.no_grad()
    def predict_soft(self, x: torch.Tensor, tta: bool = False) -> dict[str, torch.Tensor]:
        """Soft routing: P(class) = P(defect) * P(family) * P(class | family).

        A wrong stage-2 guess can still be outvoted by the other stages, but every
        fine CNN runs on every image, so there is no early exit and it is slower.
        Assumes the healthy class is global index 0 and families follow it in order.
        """
        n = x.shape[0]
        p_def = self._probs(self.triage, x, tta)[:, 1]
        p_fam = self._probs(self.family, x, tta)
        total = 1 + sum(m.head[-1].out_features for m in self.fine.values())
        probs = torch.zeros(n, total, device=x.device)
        probs[:, self.healthy_index] = 1 - p_def
        for f_i, fam in enumerate(self.family_names):
            pf = self._probs(self.fine[fam], x, tta)
            o = self.class_offsets[fam]
            probs[:, o:o + pf.shape[1]] = (p_def * p_fam[:, f_i])[:, None] * pf
        return {"pred": probs.argmax(1), "probs": probs}
