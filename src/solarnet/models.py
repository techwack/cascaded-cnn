"""Stage CNNs and the full 3-stage cascade."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageCNN(nn.Module):
    """[Conv3x3-BN-ReLU-MaxPool2] x len(channels) -> GlobalAvgPool -> Dropout -> FC.

    With the native 40x24 input, three blocks leave a 5x3 map before pooling.
    """

    def __init__(self, channels: list[int], num_classes: int, dropout: float = 0.0, in_ch: int = 1):
        super().__init__()
        layers, prev = [], in_ch
        for ch in channels:
            layers += [
                nn.Conv2d(prev, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            prev = ch
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

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Route a batch through the cascade.

        Returns global class predictions plus the per-stage decisions, so
        evaluation can attribute errors to the stage that made them.
        """
        n = x.shape[0]
        pred = torch.full((n,), self.healthy_index, dtype=torch.long, device=x.device)
        fam_pred = torch.full((n,), -1, dtype=torch.long, device=x.device)

        is_defect = self.triage(x).argmax(1) == 1
        idx = is_defect.nonzero(as_tuple=True)[0]
        if idx.numel():  # early exit: healthy panels never reach stage 2
            fam_pred[idx] = self.family(x[idx]).argmax(1)
            for f_i, fam in enumerate(self.family_names):
                sub = idx[fam_pred[idx] == f_i]
                if sub.numel():
                    pred[sub] = self.class_offsets[fam] + self.fine[fam](x[sub]).argmax(1)
        return {"pred": pred, "is_defect": is_defect, "family": fam_pred}
