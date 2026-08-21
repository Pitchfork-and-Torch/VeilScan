"""Run detectors, optionally tiled, then fuse."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from veilscan.config import VeilConfig, resolve_device
from veilscan.ensemble import fuse
from veilscan.image_io import tiles
from veilscan.registry import ensure_loaded, select
from veilscan.types import AnalyzeContext, DetectionResult, EnsembleResult


def analyze_image(
    rgb: np.ndarray,
    cfg: VeilConfig,
    detector_names: Iterable[str] | None = None,
    reference_images: list[np.ndarray] | None = None,
) -> EnsembleResult:
    ensure_loaded()
    device = resolve_device(cfg.device)
    ctx = AnalyzeContext(
        device=device,
        reference_images=reference_images,
        checkpoint_dir=cfg.checkpoint_dir,
    )
    dets = select(detector_names, tier=cfg.tier)
    h, w = rgb.shape[:2]
    pieces = tiles(rgb, cfg.tile_size, cfg.tile_overlap)

    if len(pieces) == 1:
        results = [_safe(d, rgb, ctx) for d in dets]
        return fuse(results, cfg.weights, cfg.threshold, rgb.shape)

    # Per-detector: take the max tile score (a mark in one tile is enough).
    # Heatmaps stitched by overlap-average.
    acc: dict[str, list[DetectionResult]] = {d.name: [] for d in dets}
    hm_acc: dict[str, np.ndarray] = {}
    hm_cnt: dict[str, np.ndarray] = {}
    for y1, x1, y2, x2, crop in pieces:
        for d in dets:
            r = _safe(d, crop, ctx)
            acc[d.name].append(r)
            if r.heatmap is not None:
                if d.name not in hm_acc:
                    hm_acc[d.name] = np.zeros((h, w), dtype=np.float64)
                    hm_cnt[d.name] = np.zeros((h, w), dtype=np.float64)
                hm = r.heatmap
                if hm.shape != crop.shape[:2]:
                    import cv2

                    hm = cv2.resize(hm.astype(np.float32), (crop.shape[1], crop.shape[0]))
                hm_acc[d.name][y1:y2, x1:x2] += hm
                hm_cnt[d.name][y1:y2, x1:x2] += 1.0

    merged: list[DetectionResult] = []
    for d in dets:
        rs = acc[d.name]
        if not rs:
            continue
        live = [r for r in rs if not r.skipped]
        if not live:
            merged.append(rs[0])
            continue
        best = max(live, key=lambda r: r.score)
        if d.name in hm_acc:
            best.heatmap = hm_acc[d.name] / np.maximum(hm_cnt[d.name], 1.0)
        extras = dict(best.extras)
        extras["tile_max"] = best.score
        extras["tile_mean"] = float(np.mean([r.score for r in live]))
        best.extras = extras
        merged.append(best)
    return fuse(merged, cfg.weights, cfg.threshold, rgb.shape)


def _safe(detector, rgb: np.ndarray, ctx: AnalyzeContext) -> DetectionResult:
    try:
        return detector.analyze(rgb, ctx).clamp()
    except Exception as e:
        return detector.skip(f"crashed: {type(e).__name__}: {e}")
