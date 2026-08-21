"""Weighted fusion with disagreement as uncertainty."""

from __future__ import annotations

import numpy as np

from veilscan.types import DetectionResult, EnsembleResult


DEFAULT_PEAK_OK = (
    "chi_square",
    "rs_analysis",
    "bitplane",
    "dct",
    "dwt",
    "hybrid_dds",
    "tree_ring_spectral",
    "residual_cnn",
    "fsnet_lite",
)

DEFAULT_MIX = {"full_mean": 0.20, "ok_mean": 0.20, "top_mean": 0.20, "peak": 0.40, "top_k": 3, "uncertainty": "peak_ok"}


def fuse(
    results: list[DetectionResult],
    weights: dict[str, float],
    threshold: float,
    image_shape: tuple[int, ...],
    peak_ok: list[str] | tuple[str, ...] | None = None,
    mix: dict | None = None,
) -> EnsembleResult:
    active = [r for r in results if not r.skipped and r.confidence > 0.05]
    skipped = len(results) - len(active)
    if not active:
        return EnsembleResult(
            present=False,
            score=0.0,
            confidence=0.0,
            uncertainty=1.0,
            threshold=threshold,
            explanation="No detector produced a usable score.",
            detectors=results,
            image_shape=image_shape,
            active=0,
            skipped=skipped,
        )

    w = np.array([max(weights.get(r.detector, 0.5), 0.0) * r.confidence for r in active], dtype=np.float64)
    s = np.array([r.score for r in active], dtype=np.float64)
    if w.sum() <= 1e-12:
        w = np.ones_like(s)
    full_mean = float(np.sum(w * s) / w.sum())
    allow = set(peak_ok) if peak_ok is not None else set(DEFAULT_PEAK_OK)
    mix = {**DEFAULT_MIX, **(mix or {})}
    ok_idx = [i for i, r in enumerate(active) if r.detector in allow]
    if ok_idx:
        s_ok = s[ok_idx]
        w_ok = w[ok_idx]
        peak = float(np.max(s_ok))
        order = np.argsort(-(s_ok * w_ok))
        k = min(int(mix.get("top_k", 3)), s_ok.size)
        top = order[:k]
        top_mean = float(np.sum(w_ok[top] * s_ok[top]) / max(w_ok[top].sum(), 1e-12))
        ok_mean = float(np.sum(w_ok * s_ok) / max(w_ok.sum(), 1e-12))
        unc_src = s_ok if mix.get("uncertainty", "peak_ok") == "peak_ok" else s
    else:
        peak = float(np.max(s))
        top_mean = full_mean
        ok_mean = full_mean
        unc_src = s
    score = (
        float(mix.get("full_mean", 0.20)) * full_mean
        + float(mix.get("ok_mean", 0.20)) * ok_mean
        + float(mix.get("top_mean", 0.20)) * top_mean
        + float(mix.get("peak", 0.40)) * peak
    )
    uncertainty = float(np.std(unc_src)) if unc_src.size > 1 else 0.0
    coverage = min(1.0, len(active) / 8.0)
    confidence = float(np.clip((1.0 - uncertainty) * (0.5 + 0.5 * coverage), 0.0, 1.0))
    present = score >= threshold

    heatmap = _merge_heatmaps(active, image_shape)

    primary = [r for r in active if r.detector in allow]
    rank_src = primary or active
    top = sorted(rank_src, key=lambda r: r.score * weights.get(r.detector, 0.5), reverse=True)[:4]
    bits = ", ".join(f"{r.detector}={r.score:.2f}" for r in top)
    if present:
        expl = (
            f"Ensemble score {score:.3f} >= {threshold:.2f} (watermark likely). "
            f"Uncertainty {uncertainty:.3f}. Top: {bits}."
        )
    else:
        expl = (
            f"Ensemble score {score:.3f} < {threshold:.2f} (no strong invisible-watermark evidence). "
            f"Uncertainty {uncertainty:.3f}. Top: {bits}."
        )
    return EnsembleResult(
        present=present,
        score=score,
        confidence=confidence,
        uncertainty=uncertainty,
        threshold=threshold,
        explanation=expl,
        detectors=results,
        heatmap=heatmap,
        image_shape=image_shape,
        active=len(active),
        skipped=skipped,
    )


def _merge_heatmaps(active: list[DetectionResult], image_shape: tuple[int, ...]) -> np.ndarray | None:
    maps = [r.heatmap for r in active if r.heatmap is not None]
    if not maps or len(image_shape) < 2:
        return None
    h, w = int(image_shape[0]), int(image_shape[1])
    acc = np.zeros((h, w), dtype=np.float64)
    n = 0
    for m in maps:
        if m.ndim != 2:
            continue
        mm = _resize(m, h, w)
        acc += mm
        n += 1
    if n == 0:
        return None
    acc /= n
    mx = acc.max()
    if mx > 0:
        acc = acc / mx
    return acc


def _resize(m: np.ndarray, h: int, w: int) -> np.ndarray:
    if m.shape == (h, w):
        return m.astype(np.float64)
    try:
        import cv2

        return cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float64)
    except Exception:
        ys = (np.linspace(0, m.shape[0] - 1, h)).astype(int)
        xs = (np.linspace(0, m.shape[1] - 1, w)).astype(int)
        return m[ys][:, xs].astype(np.float64)
