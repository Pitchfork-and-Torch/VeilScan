"""Run detectors, optionally tiled, then fuse."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from veilscan.config import (
    VeilConfig,
    load_camera_operating_point,
    load_operating_point,
    merge_operating_point,
    resolve_device,
)
from veilscan.dsp import jpeg_blockiness, jpeg_like
from veilscan.ensemble import fuse
from veilscan.image_io import tiles
from veilscan.jpeg_meta import inspect_jpeg, jpeg_freq_weight
from veilscan.registry import ensure_loaded, select
from veilscan.types import AnalyzeContext, DetectionResult, EnsembleResult


def analyze_image(
    rgb: np.ndarray,
    cfg: VeilConfig,
    detector_names: Iterable[str] | None = None,
    reference_images: list[np.ndarray] | None = None,
    *,
    jpeg_container: bool | None = None,
    source_bytes: bytes | None = None,
    jpeg_quality_est: int | None = None,
) -> EnsembleResult:
    ensure_loaded()
    device = resolve_device(cfg.device)
    ctx = AnalyzeContext(
        device=device,
        reference_images=reference_images,
        checkpoint_dir=cfg.checkpoint_dir,
        extra={
            "jpeg_quality_probe": cfg.jpeg_quality_probe,
            "peak_ok": list(cfg.peak_ok),
            "jpeg_container": bool(jpeg_container),
        },
    )
    dets = select(detector_names, tier=cfg.tier)
    h, w = rgb.shape[:2]
    pieces = tiles(rgb, cfg.tile_size, cfg.tile_overlap)

    if len(pieces) == 1:
        results = [_safe(d, rgb, ctx) for d in dets]
        return _fuse_calibrated(
            results,
            cfg,
            rgb.shape,
            rgb,
            jpeg_container=jpeg_container,
            source_bytes=source_bytes,
            jpeg_quality_est=jpeg_quality_est,
        )

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
    return _fuse_calibrated(
        merged,
        cfg,
        rgb.shape,
        rgb,
        jpeg_container=jpeg_container,
        source_bytes=source_bytes,
        jpeg_quality_est=jpeg_quality_est,
    )


def _fuse_calibrated(
    results: list,
    cfg: VeilConfig,
    shape: tuple[int, ...],
    rgb: np.ndarray | None = None,
    jpeg_container: bool | None = None,
    source_bytes: bytes | None = None,
    jpeg_quality_est: int | None = None,
) -> EnsembleResult:
    op = load_operating_point(cfg.operating_point_path)
    mix = merge_operating_point(cfg.fusion, op)
    threshold = float(cfg.threshold)
    if op and str(op.get("status") or "") == "locked" and op.get("threshold") is not None:
        if mix.get("mode") != "specialist_or":
            threshold = float(op["threshold"])
    jpeg_info = inspect_jpeg(source_bytes)
    if jpeg_quality_est is not None:
        jpeg_info["quality_est"] = int(jpeg_quality_est)
    container = bool(jpeg_info.get("container")) or bool(jpeg_container)
    blockiness = 0.0
    if rgb is not None:
        blockiness = jpeg_blockiness(rgb)
    like = bool(container or (rgb is not None and jpeg_like(rgb)))
    mix["jpeg_like"] = like
    mix["jpeg_freq_weight"] = jpeg_freq_weight(
        jpeg_info.get("quality_est"),
        jpeg_like=like,
        blockiness=blockiness,
    )
    if cfg.apply_calibration:
        from veilscan.calibrate import apply_affine, load_calibration

        cal = load_calibration(cfg.calibration_path)
        heads = cal.get("detectors") or {}
        for r in results:
            if not r.skipped:
                r.score = apply_affine(heads.get(r.detector), r.score)
                r.clamp()
        fused = fuse(results, cfg.weights, threshold, shape, peak_ok=cfg.peak_ok, mix=mix)
        fused.score = apply_affine(cal.get("ensemble"), fused.score)
        fused.score = float(np.clip(fused.score, 0.0, 1.0))
        if mix.get("mode") != "specialist_or":
            fused.present = fused.score >= fused.threshold
        _stamp_jpeg(fused, mix, blockiness, container, jpeg_info)
        return _attach_camera(fused, cfg)
    fused = fuse(results, cfg.weights, threshold, shape, peak_ok=cfg.peak_ok, mix=mix)
    _stamp_jpeg(fused, mix, blockiness, container, jpeg_info)
    return _attach_camera(fused, cfg)


def _stamp_jpeg(fused: EnsembleResult, mix: dict, blockiness: float, container: bool, info: dict) -> None:
    fused.jpeg_blockiness = float(blockiness)
    fused.jpeg_like = bool(mix.get("jpeg_like"))
    fused.jpeg_container = bool(container)
    q = info.get("quality_est")
    fused.jpeg_quality_est = int(q) if q is not None else None
    sub = info.get("subsampling")
    fused.jpeg_subsampling = str(sub) if sub else None
    fused.jpeg_freq_weight = float(mix.get("jpeg_freq_weight") or 0.0)
    q34 = info.get("luma_q_34")
    q43 = info.get("luma_q_43")
    fused.jpeg_luma_q_34 = int(q34) if q34 is not None else None
    fused.jpeg_luma_q_43 = int(q43) if q43 is not None else None
    pair = None
    for d in fused.detectors:
        if d.detector == "dct" and not d.skipped:
            raw = (d.extras or {}).get("pair_34_43")
            if raw is not None:
                pair = float(raw)
                break
    fused.jpeg_dct_pair_34_43 = pair


def apply_present_policy(fused: EnsembleResult, policy: str | None = None) -> EnsembleResult:
    p = (policy or "generator").strip().lower()
    if p not in {"generator", "camera", "both"}:
        p = "generator"
    fused.policy = p
    if p == "camera" and fused.camera:
        fused.present = bool(fused.camera.get("present"))
        fused.threshold = float(fused.camera["threshold"])
        oid = fused.camera.get("operating_point_id")
        if oid:
            fused.operating_point_id = str(oid)
    return fused


def _attach_camera(fused: EnsembleResult, cfg: VeilConfig) -> EnsembleResult:
    cam = load_camera_operating_point()
    if not cam or str(cam.get("status") or "") not in {"provisional", "locked"}:
        return fused
    t = cam.get("threshold")
    if t is None:
        return fused
    fused.camera = {
        "operating_point_id": cam.get("id"),
        "status": cam.get("status"),
        "threshold": float(t),
        "present": float(fused.score) >= float(t),
        "fpr_est": cam.get("fpr_est"),
        "n": cam.get("n"),
    }
    return fused


def _safe(detector, rgb: np.ndarray, ctx: AnalyzeContext) -> DetectionResult:
    try:
        return detector.analyze(rgb, ctx).clamp()
    except Exception as e:
        return detector.skip(f"crashed: {type(e).__name__}: {e}")
