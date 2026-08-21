"""Blind LSB hotspot hunt: tiles with decorrelated bit-0 planes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hotspot:
    x: int
    y: int
    w: int
    h: int
    score: float
    corr: float

    def pad(self, rgb: np.ndarray, margin: int = 16) -> "Hotspot":
        h, w = rgb.shape[:2]
        x0 = max(0, self.x - margin)
        y0 = max(0, self.y - margin)
        x1 = min(w, self.x + self.w + margin)
        y1 = min(h, self.y + self.h + margin)
        return Hotspot(x0, y0, x1 - x0, y1 - y0, self.score, self.corr)

    def crop(self, rgb: np.ndarray) -> np.ndarray:
        return rgb[self.y : self.y + self.h, self.x : self.x + self.w]

    def to_json(self) -> dict:
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "score": round(float(self.score), 6),
            "corr": round(float(self.corr), 6),
        }


def cluster_hotspots(spots: list[Hotspot], dist: float = 110.0) -> list[list[Hotspot]]:
    """Group nearby tiles so the report can draw one CLUSTER hull."""
    groups: list[list[Hotspot]] = []
    for h in spots:
        cx, cy = h.x + h.w / 2.0, h.y + h.h / 2.0
        placed = False
        for g in groups:
            gx = sum(i.x + i.w / 2.0 for i in g) / len(g)
            gy = sum(i.y + i.h / 2.0 for i in g) / len(g)
            if (cx - gx) ** 2 + (cy - gy) ** 2 <= dist * dist:
                g.append(h)
                placed = True
                break
        if not placed:
            groups.append([h])
    return groups


def lsb_autocorr(plane: np.ndarray) -> float:
    a = np.asarray(plane, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 8:
        return 1.0
    left = a[:, :-1].ravel()
    right = a[:, 1:].ravel()
    if left.size < 64 or left.std() < 1e-9 or right.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(left, right)[0, 1])


def _iou(a: Hotspot, b: Hotspot) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union else 0.0


def _nms(cands: list[Hotspot], iou_max: float = 0.45, keep: int = 8) -> list[Hotspot]:
    ordered = sorted(cands, key=lambda h: -h.score)
    kept: list[Hotspot] = []
    for h in ordered:
        if any(_iou(h, k) > iou_max for k in kept):
            continue
        kept.append(h)
        if len(kept) >= keep:
            break
    return kept


PATCH_SIZES = (
    (70, 42),
    (64, 48),
    (48, 32),
    (80, 40),
    (32, 32),
    (96, 64),
    (64, 64),
)


def iter_patch_windows(rgb: np.ndarray, hs: Hotspot, margin: int = 24):
    """Rectangles inside a hotspot. Patch LSB is sequential only inside its own box."""
    pad = hs.pad(rgb, margin)
    yield hs
    yield pad
    for pw, ph in PATCH_SIZES:
        if pad.w < pw or pad.h < ph:
            continue
        x_step = 2
        y_step = 2
        ymax = pad.y + pad.h - ph
        xmax = pad.x + pad.w - pw
        for y in range(pad.y, ymax + 1, y_step):
            for x in range(pad.x, xmax + 1, x_step):
                yield Hotspot(x, y, pw, ph, hs.score, hs.corr)


def find_lsb_hotspots(
    rgb: np.ndarray,
    tile: int = 64,
    step: int = 16,
    keep: int = 12,
) -> list[Hotspot]:
    """Lowest LSB autocorrelation tiles, NMS-merged. No coordinates required."""
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] < 3:
        return []
    h, w = img.shape[:2]
    if min(h, w) < tile:
        plane = img[..., 0] & 1
        corr = lsb_autocorr(plane)
        return [Hotspot(0, 0, w, h, score=max(0.0, 1.0 - corr), corr=corr)]

    raw: list[Hotspot] = []
    corrs: list[float] = []
    for y in range(0, h - tile + 1, step):
        for x in range(0, w - tile + 1, step):
            scores = []
            for c in range(3):
                plane = img[y : y + tile, x : x + tile, c] & 1
                scores.append(lsb_autocorr(plane))
            corr = float(np.median(scores))
            corrs.append(corr)
            raw.append(Hotspot(x, y, tile, tile, score=max(0.0, 1.0 - corr), corr=corr))
    if not raw:
        return []
    med = float(np.median(corrs))
    thresh = min(med - 0.08, 0.18)
    filtered = [hs for hs in raw if hs.corr <= thresh]
    if len(filtered) < 3:
        filtered = sorted(raw, key=lambda hs: hs.corr)[: max(8, keep * 2)]
    return _nms(filtered, keep=keep)
