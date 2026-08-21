"""Compact residual CNN (YeNet / SRNet inspired, tiny for CPU training)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # First layer: learned high-pass (initialized near KV-like).
        self.hp = nn.Conv2d(3, 16, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.hp.weight)
        self.block = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.hp(x)
        z = self.block(z).flatten(1)
        return self.head(z).squeeze(1)
