"""FSNet-lite: ASPM + backbone + DMSA. RGB+LSB input so bit and frequency marks both survive."""

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


def stem_channels(x: torch.Tensor) -> torch.Tensor:
    """RGB + LSB planes + amplified high-pass residual (DCT/spread live here)."""
    bits = torch.round(x * 255.0) % 2.0
    blur = F.avg_pool2d(x, 5, stride=1, padding=2)
    hp = (x - blur) * 16.0
    return torch.cat([x, bits, hp], dim=1)


class ASPM(nn.Module):
    def __init__(self, size: int = 64, in_ch: int = 9) -> None:
        super().__init__()
        # Bias the gate toward higher frequencies (center of shifted DCT is DC).
        yy = torch.linspace(-1.0, 1.0, size).view(1, 1, size, 1)
        xx = torch.linspace(-1.0, 1.0, size).view(1, 1, 1, size)
        high = (yy ** 2 + xx ** 2).clamp(0, 1) - 0.35
        self.gate = nn.Parameter(high.expand(1, in_ch, size, size).contiguous())
        self.fuse = nn.Conv2d(in_ch * 3, 32, 3, padding=1)
        self.fuse_bn = nn.BatchNorm2d(32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = dct2(x)
        mask = torch.sigmoid(self.gate)
        mask = F.interpolate(mask, size=freq.shape[-2:], mode="bilinear", align_corners=False)
        residual = idct2(freq * mask)
        mx = F.max_pool2d(residual, 3, stride=1, padding=1)
        fused = torch.cat([x, residual, mx], dim=1)
        return F.leaky_relu(self.fuse_bn(self.fuse(fused)), 0.1)


class DMSA(nn.Module):
    def __init__(self, channels: int, k: int = 8) -> None:
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        uv = []
        for i in range(k):
            uv.append((2 + i // 3, 2 + i % 3))
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
        self.aspm = ASPM(64, in_ch=9)
        self.backbone = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.dmsa = DMSA(128, k=8)
        self.head = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.aspm(stem_channels(x))
        z = self.backbone(z)
        z = self.dmsa(z)
        mean = F.adaptive_avg_pool2d(z, 1).flatten(1)
        var = F.adaptive_avg_pool2d(z * z, 1).flatten(1) - mean * mean
        std = torch.sqrt(var.clamp_min(1e-6))
        return self.head(torch.cat([mean, std], dim=1)).squeeze(1)
