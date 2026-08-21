"""FSNet-lite: ASPM + small backbone + DMSA (Ao et al. AWPD ideas, independent reimplementation)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dct_1d(x: torch.Tensor, dim: int) -> torch.Tensor:
    n = x.size(dim)
    x = x.transpose(dim, -1)
    v = torch.cat([x, x.flip(-1)], dim=-1)
    V = torch.fft.fft(v, dim=-1).real[..., :n]
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    V = V * (2.0 * torch.cos(math.pi * k / (2.0 * n)))
    return V.transpose(-1, dim)


def dct2(x: torch.Tensor) -> torch.Tensor:
    return _dct_1d(_dct_1d(x, -2), -1)


def idct2(x: torch.Tensor) -> torch.Tensor:
    # Pair with the unnormalized type-II used above via IFFT even extension.
    n0 = x.size(-2)
    n1 = x.size(-1)
    k0 = torch.arange(n0, device=x.device, dtype=x.dtype).view(1, 1, n0, 1)
    k1 = torch.arange(n1, device=x.device, dtype=x.dtype).view(1, 1, 1, n1)
    y = x / (2.0 * torch.cos(math.pi * k0 / (2.0 * n0)).clamp_min(1e-6))
    y = y / (2.0 * torch.cos(math.pi * k1 / (2.0 * n1)).clamp_min(1e-6))
    v = torch.cat([y, y.flip(-2)], dim=-2)
    s = torch.fft.ifft(v, dim=-2).real[..., :n0, :]
    v2 = torch.cat([s, s.flip(-1)], dim=-1)
    return torch.fft.ifft(v2, dim=-1).real[..., :n1]


class ASPM(nn.Module):
    def __init__(self, size: int = 64) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1, 1, size, size))
        self.fuse = nn.Conv2d(3, 32, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,3,H,W in [0,1]
        gray = x.mean(dim=1, keepdim=True)
        freq = dct2(gray)
        mask = torch.sigmoid(self.gate)
        mask = F.interpolate(mask, size=freq.shape[-2:], mode="bilinear", align_corners=False)
        spatial = idct2(freq * mask)
        residual = spatial
        mx = F.max_pool2d(residual, 3, stride=1, padding=1)
        fused = torch.cat([gray, residual, mx], dim=1)
        return self.fuse(fused)


class DMSA(nn.Module):
    def __init__(self, channels: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        # Predefined (u,v) pairs emphasizing higher frequencies.
        uv = []
        for i in range(k):
            uv.append((1 + i // 3, 1 + i % 3))
        self.register_buffer("uv", torch.tensor(uv, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        device = x.device
        yy = torch.linspace(0, h - 1, h, device=device).view(1, 1, h, 1)
        xx = torch.linspace(0, w - 1, w, device=device).view(1, 1, 1, w)
        feats = []
        for i in range(self.k):
            u, v = self.uv[i]
            basis = torch.cos(math.pi * yy / h * (u + 0.5)) * torch.cos(math.pi * xx / w * (v + 0.5))
            proj = x * basis
            avg = proj.mean(dim=(-2, -1))
            mx = proj.amax(dim=(-2, -1))
            mn = proj.amin(dim=(-2, -1))
            feats.append((avg + mx + mn) / 3.0)
        desc = torch.stack(feats, dim=0).mean(dim=0)
        wch = self.mlp(desc).unsqueeze(-1).unsqueeze(-1)
        return x * wch


class FSNetLite(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.aspm = ASPM(64)
        self.backbone = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dmsa = DMSA(128, k=8)
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.aspm(x)
        z = self.backbone(z)
        z = self.dmsa(z)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.head(z).squeeze(1)
