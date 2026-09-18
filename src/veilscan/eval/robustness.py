"""Attack suite for robustness evaluation."""

from __future__ import annotations

import numpy as np

import cv2

from veilscan.image_io import jpeg_roundtrip


def apply_attack(rgb: np.ndarray, name: str, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    if name == "identity":
        return rgb
    if name.startswith("jpeg"):
        q = int(name.split("_")[1]) if "_" in name else 75
        return jpeg_roundtrip(rgb, q)
    if name == "resize_90":
        h, w = rgb.shape[:2]
        small = cv2.resize(rgb, (int(w * 0.9), int(h * 0.9)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if name == "crop_90":
        h, w = rgb.shape[:2]
        m0, m1 = int(h * 0.05), int(w * 0.05)
        crop = rgb[m0 : h - m0, m1 : w - m1]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    if name == "noise":
        n = rng.normal(0, 4.0, size=rgb.shape)
        return np.clip(rgb.astype(np.float64) + n, 0, 255).astype(np.uint8)
    if name == "blur":
        return cv2.GaussianBlur(rgb, (5, 5), 0)
    if name == "jitter":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float64)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.08, 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * 0.96, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    raise KeyError(f"unknown attack {name}")


DEFAULT_ATTACKS = [
    "identity",
    "jpeg_90",
    "jpeg_70",
    "jpeg_50",
    "resize_90",
    "crop_90",
    "noise",
    "blur",
    "jitter",
]
