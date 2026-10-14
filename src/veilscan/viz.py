"""Heatmap overlay export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veilscan.image_io import save_rgb


def overlay(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = rgb.shape[:2]
    hm = heatmap.astype(np.float64)
    if hm.shape != (h, w):
        import cv2

        hm = cv2.resize(hm, (w, h), interpolation=cv2.INTER_LINEAR)
    hm = np.clip(hm, 0, 1)
    color = np.zeros_like(rgb, dtype=np.float64)
    color[..., 0] = 255.0 * hm
    color[..., 1] = 40.0 * hm
    color[..., 2] = 20.0 * (1.0 - hm)
    out = (1.0 - alpha * hm[..., None]) * rgb.astype(np.float64) + (alpha * hm[..., None]) * color
    return np.clip(out, 0, 255).astype(np.uint8)


def save_overlay(path: str | Path, rgb: np.ndarray, heatmap: np.ndarray | None) -> None:
    if heatmap is None:
        save_rgb(path, rgb)
        return
    save_rgb(path, overlay(rgb, heatmap))
