"""Compact residual CNN. Concat 0-1 RGB with explicit LSB planes so bit payloads are visible."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 3 RGB + 3 LSB bit planes
        self.hp = nn.Conv2d(6, 16, 5, padding=2, bias=False)
        nn.init.xavier_uniform_(self.hp.weight)
        self.hp_bn = nn.BatchNorm2d(16)
        self.block = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.head = nn.Linear(192, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bits = torch.round(x * 255.0) % 2.0
        z = self.hp_bn(self.hp(torch.cat([x, bits], dim=1)))
        z = self.block(z)
        mean = F.adaptive_avg_pool2d(z, 1).flatten(1)
        var = F.adaptive_avg_pool2d(z * z, 1).flatten(1) - mean * mean
        std = torch.sqrt(var.clamp_min(1e-6))
        return self.head(torch.cat([mean, std], dim=1)).squeeze(1)
