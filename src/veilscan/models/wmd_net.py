"""Small conv stack used as the WMD offset-learning body."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WMDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        ch = 32
        self.features = nn.Sequential(
            nn.Conv2d(3, ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(ch * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.head(z).squeeze(1)
