"""Local foundation-style anomaly: patch PCA reconstruction. No mandatory download."""

from __future__ import annotations

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.dsp import score_from_stat, to_gray
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult


class FoundationDetector(BaseDetector):
    name = "foundation"
    tier = "foundation"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        err, hm = _patch_pca_residual(gray, patch=8, stride=8, k=12)
        score = score_from_stat(err, center=18.0, scale=12.0)
        extras = {"pca_recon_mse": err}
        expl = (
            f"Patch-PCA reconstruction residual MSE={err:.3f}. "
            "Local stand-in for VFM anomaly; DINOv2/CLIP not downloaded by default."
        )
        return DetectionResult(self.name, float(score), 0.5, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


def _patch_pca_residual(gray: np.ndarray, patch: int, stride: int, k: int) -> tuple[float, np.ndarray]:
    h, w = gray.shape
    patches = []
    coords = []
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            patches.append(gray[y : y + patch, x : x + patch].reshape(-1))
            coords.append((y, x))
    if len(patches) < k + 2:
        return 0.0, np.zeros_like(gray)
    X = np.stack(patches, axis=0)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # economy SVD
    try:
        _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0, np.zeros_like(gray)
    k = min(k, vt.shape[0])
    basis = vt[:k]
    recon = (Xc @ basis.T) @ basis + mean
    mse = np.mean((X - recon) ** 2, axis=1)
    hm = np.zeros_like(gray, dtype=np.float64)
    cnt = np.zeros_like(gray, dtype=np.float64)
    for (y, x), e in zip(coords, mse):
        hm[y : y + patch, x : x + patch] += e
        cnt[y : y + patch, x : x + patch] += 1.0
    hm = hm / np.maximum(cnt, 1.0)
    hm = hm / (np.percentile(hm, 99) + 1e-9)
    return float(np.mean(mse)), np.clip(hm, 0, 1)


def register_foundation() -> None:
    register(FoundationDetector())
