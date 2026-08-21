"""Residual, color-space, reconstruction, and higher-order detectors."""

from __future__ import annotations

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.dsp import (
    entropy,
    highpass_residual,
    kurtosis,
    kv_residual,
    rgb_to_hsv,
    rgb_to_ycbcr,
    score_from_stat,
    skewness,
    to_gray,
)
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult


class SRMDetector(BaseDetector):
    name = "srm"
    tier = "residual"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        kv = kv_residual(gray)
        hp = highpass_residual(gray, 5)
        q = np.clip(np.rint(kv / 4.0), -4, 4).astype(np.int8)
        cooc = _cooc_h(q, lo=-4, hi=4)
        # Peakiness of residual co-occurrence vs a spread residual.
        p = cooc / (cooc.sum() + 1e-12)
        cooc_ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
        k = kurtosis(kv)
        e = float(np.mean(hp ** 2))
        score = (
            0.4 * score_from_stat(k, center=12.0, scale=10.0, invert=True)
            + 0.3 * score_from_stat(cooc_ent, center=5.4, scale=0.8, invert=True)
            + 0.3 * score_from_stat(np.log1p(e), center=4.5, scale=1.6)
        )
        extras = {"kv_kurtosis": k, "cooc_entropy": cooc_ent, "hp_energy": e}
        hm = np.clip(np.abs(kv) / (np.percentile(np.abs(kv), 99) + 1e-9), 0, 1)
        expl = (
            f"SRM-lite (KV residual + co-occurrence). kurtosis={k:.2f}, "
            f"cooc entropy={cooc_ent:.2f}."
        )
        return DetectionResult(self.name, float(score), 0.75, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


def _cooc_h(q: np.ndarray, lo: int, hi: int) -> np.ndarray:
    a = q[:, :-1]
    b = q[:, 1:]
    n = hi - lo + 1
    aa = (a - lo).ravel()
    bb = (b - lo).ravel()
    ok = (aa >= 0) & (aa < n) & (bb >= 0) & (bb < n)
    return np.bincount(aa[ok] * n + bb[ok], minlength=n * n).astype(np.float64)


class ColorSpaceDetector(BaseDetector):
    name = "color_spaces"
    tier = "residual"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        if image.ndim != 3 or image.shape[2] < 3:
            return self.skip("need RGB")
        y, cb, cr = rgb_to_ycbcr(image)
        _, s, v = rgb_to_hsv(image)
        y_hp = highpass_residual(y, 5)
        cb_hp = highpass_residual(cb, 5)
        cr_hp = highpass_residual(cr, 5)
        chroma = float(np.mean(cb_hp ** 2) + np.mean(cr_hp ** 2))
        luma = float(np.mean(y_hp ** 2) + 1e-12)
        ratio = chroma / luma
        s_ent = entropy(s, bins=64)
        score = 0.6 * score_from_stat(ratio, center=0.35, scale=0.25) + 0.4 * score_from_stat(
            s_ent, center=5.2, scale=1.0
        )
        extras = {"chroma_luma_hp": ratio, "sat_entropy": s_ent}
        hm = np.clip((np.abs(cb_hp) + np.abs(cr_hp)) / (np.percentile(np.abs(cb_hp) + np.abs(cr_hp), 99) + 1e-9), 0, 1)
        expl = f"Color-space residuals. chroma/luma HP energy={ratio:.3f}."
        return DetectionResult(self.name, float(score), 0.55, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


class ReconstructionDetector(BaseDetector):
    name = "reconstruction"
    tier = "residual"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        hp = highpass_residual(gray, 7)
        # Wiener-ish: residual energy after a second, stronger blur should be tiny for natural
        # 1/f content and larger if a high-frequency pattern was added.
        hp2 = highpass_residual(gray, 15)
        e1 = float(np.mean(hp ** 2) + 1e-12)
        e2 = float(np.mean(hp2 ** 2) + 1e-12)
        ratio = e2 / e1
        k = kurtosis(hp)
        score = 0.5 * score_from_stat(np.log1p(e1), center=4.0, scale=1.5) + 0.5 * score_from_stat(
            k, center=10.0, scale=8.0, invert=True
        )
        extras = {"hp_energy": e1, "wide_narrow_ratio": ratio, "hp_kurtosis": k}
        hm = np.clip(np.abs(hp) / (np.percentile(np.abs(hp), 99) + 1e-9), 0, 1)
        expl = f"Denoising reconstruction residual. HP energy={e1:.3f}, kurtosis={k:.2f}."
        return DetectionResult(self.name, float(score), 0.6, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


class HigherOrderDetector(BaseDetector):
    name = "higher_order"
    tier = "residual"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        hp = highpass_residual(gray, 5)
        sk = skewness(hp)
        ku = kurtosis(hp)
        # Cheap 1D bispectral proxy: correlation of residual with its square along rows.
        row = hp.mean(axis=0)
        row = row - row.mean()
        if row.size < 32:
            return self.skip("too small")
        b = float(np.corrcoef(row[:-1], (row[1:] ** 2))[0, 1]) if np.std(row) > 1e-9 else 0.0
        if not np.isfinite(b):
            b = 0.0
        score = 0.4 * score_from_stat(abs(sk), center=0.35, scale=0.35) + 0.3 * score_from_stat(
            abs(ku - 3.0), center=6.0, scale=6.0
        ) + 0.3 * score_from_stat(abs(b), center=0.08, scale=0.08)
        extras = {"skew": sk, "kurtosis": ku, "row_bispec_proxy": b}
        expl = f"Higher-order residual stats. skew={sk:.3f}, kurtosis={ku:.2f}, bispec-proxy={b:.3f}."
        return DetectionResult(self.name, float(score), 0.5, expl, extras=extras, tier=self.tier).clamp()


def register_residual() -> None:
    register(SRMDetector())
    register(ColorSpaceDetector())
    register(ReconstructionDetector())
    register(HigherOrderDetector())
