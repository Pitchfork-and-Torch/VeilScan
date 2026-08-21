"""Public Python API."""

from __future__ import annotations

from pathlib import Path

from veilscan.config import VeilConfig
from veilscan.engine import analyze_image
from veilscan.image_io import load_rgb
from veilscan.registry import all_detectors, ensure_loaded
from veilscan.types import EnsembleResult


def analyze(
    image,
    *,
    config: VeilConfig | None = None,
    detectors: list[str] | None = None,
    reference_images=None,
) -> EnsembleResult:
    cfg = config or VeilConfig.load()
    return analyze_image(image, cfg, detectors, reference_images)


def analyze_path(
    path: str | Path,
    *,
    config: VeilConfig | None = None,
    detectors: list[str] | None = None,
    reference_dir: str | Path | None = None,
) -> EnsembleResult:
    rgb = load_rgb(path)
    refs = None
    if reference_dir:
        from veilscan.image_io import iter_images

        refs = [load_rgb(p) for p in iter_images(reference_dir)[:32]]
    return analyze(rgb, config=config, detectors=detectors, reference_images=refs)


def analyze_images(
    images,
    *,
    config: VeilConfig | None = None,
    detectors: list[str] | None = None,
    reference_images=None,
) -> list[EnsembleResult]:
    """Sequential batch. GPU minibatching is Phase 4 follow-up after checkpoints exist."""
    cfg = config or VeilConfig.load()
    return [analyze_image(im, cfg, detectors, reference_images) for im in images]


def list_detectors() -> list[dict[str, str]]:
    ensure_loaded()
    return [{"name": d.name, "tier": d.tier} for d in all_detectors()]
