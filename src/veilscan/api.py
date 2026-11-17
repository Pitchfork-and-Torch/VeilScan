"""Public Python API."""

from __future__ import annotations

from pathlib import Path

from veilscan.config import VeilConfig
from veilscan.engine import analyze_image, apply_present_policy
from veilscan.image_io import load_rgb, load_rgb_bytes
from veilscan.registry import all_detectors, ensure_loaded
from veilscan.types import EnsembleResult


def analyze(
    image,
    *,
    config: VeilConfig | None = None,
    detectors: list[str] | None = None,
    reference_images=None,
    jpeg_container: bool | None = None,
    source_bytes: bytes | None = None,
) -> EnsembleResult:
    cfg = config or VeilConfig.load()
    return analyze_image(
        image,
        cfg,
        detectors,
        reference_images,
        jpeg_container=jpeg_container,
        source_bytes=source_bytes,
    )


def analyze_path(
    path: str | Path,
    *,
    config: VeilConfig | None = None,
    detectors: list[str] | None = None,
    reference_dir: str | Path | None = None,
    policy: str = "generator",
) -> EnsembleResult:
    data = Path(path).read_bytes()
    from veilscan.decode.container import sniff_kind

    kind = sniff_kind(data)
    hunt_only = {
        "pdf": "PDF is a hunt target, not an image scan. Use: veilscan hunt",
        "mp3": "MP3 is a hunt target, not an image scan. Use: veilscan hunt",
        "flac": "FLAC is a hunt target, not an image scan. Use: veilscan hunt",
    }
    if kind in hunt_only:
        raise ValueError(hunt_only[kind])
    rgb = load_rgb_bytes(data)
    refs = None
    if reference_dir:
        from veilscan.image_io import iter_images

        refs = [load_rgb(p) for p in iter_images(reference_dir)[:32]]
    result = analyze(
        rgb,
        config=config,
        detectors=detectors,
        reference_images=refs,
        source_bytes=data,
    )
    return apply_present_policy(result, policy)


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
