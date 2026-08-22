"""YAML config loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PEAK_OK = [
    "chi_square",
    "rs_analysis",
    "bitplane",
    "dct",
    "dwt",
    "hybrid_dds",
    "tree_ring_spectral",
    "residual_cnn",
    "fsnet_lite",
]

DEFAULT_FUSION = {
    "full_mean": 0.20,
    "ok_mean": 0.20,
    "top_mean": 0.20,
    "peak": 0.40,
    "top_k": 3,
    "uncertainty": "peak_ok",
    "mode": "legacy",
    "t_lsb": 0.50,
    "t_freq": 0.50,
    "t_class": 0.48,
}

DEFAULTS = {
    "threshold": 0.48,
    "tile_size": 1024,
    "tile_overlap": 64,
    "device": "auto",
    "tier": "all",
    "jpeg_quality_probe": 95,
    "apply_calibration": True,
    "calibration_path": None,
    "operating_point_path": None,
    "peak_ok": list(DEFAULT_PEAK_OK),
    "fusion": dict(DEFAULT_FUSION),
    "weights": {
        "chi_square": 1.0,
        "rs_analysis": 1.05,
        "sample_pairs": 0.95,
        "bitplane": 0.85,
        "histogram": 0.7,
        "patchwork": 0.8,
        "dct": 1.0,
        "dft": 0.75,
        "dwt": 0.95,
        "hybrid_dds": 0.8,
        "tree_ring_spectral": 0.9,
        "fourier_mellin": 0.45,
        "block_multiscale": 0.7,
        "srm": 0.9,
        "color_spaces": 0.55,
        "reconstruction": 0.6,
        "higher_order": 0.5,
        "jpeg_ela": 0.55,
        "residual_cnn": 1.15,
        "fsnet_lite": 1.2,
        "wmd": 1.1,
        "foundation": 0.55,
    },
}


def _pkg_default_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass
class VeilConfig:
    threshold: float = 0.48
    tile_size: int = 1024
    tile_overlap: int = 64
    device: str = "auto"
    tier: str = "all"
    jpeg_quality_probe: int = 95
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULTS["weights"]))
    peak_ok: list[str] = field(default_factory=lambda: list(DEFAULT_PEAK_OK))
    fusion: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FUSION))
    apply_calibration: bool = True
    calibration_path: str | None = None
    operating_point_path: str | None = None
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> VeilConfig:
        data = dict(DEFAULTS)
        weights = dict(DEFAULTS["weights"])
        fusion = dict(DEFAULT_FUSION)
        peak_ok = list(DEFAULT_PEAK_OK)
        src = Path(path) if path else _pkg_default_path()
        if src.is_file():
            with src.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            weights.update(loaded.pop("weights", {}) or {})
            fusion.update(loaded.pop("fusion", {}) or {})
            po = loaded.pop("peak_ok", None)
            if po:
                peak_ok = list(po)
            data.update(loaded)
        data["weights"] = weights
        data["fusion"] = fusion
        data["peak_ok"] = peak_ok
        known = {
            k: data[k]
            for k in (
                "threshold",
                "tile_size",
                "tile_overlap",
                "device",
                "tier",
                "jpeg_quality_probe",
                "weights",
                "peak_ok",
                "fusion",
                "apply_calibration",
                "calibration_path",
                "operating_point_path",
            )
        }
        cfg = cls(**known)
        cfg.checkpoint_dir = data.get("checkpoint_dir")
        return cfg


def _pkg_operating_point_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "operating_point.json"


def load_operating_point(path: str | Path | None = None) -> dict[str, Any] | None:
    src = Path(path) if path else _pkg_operating_point_path()
    if not src.is_file():
        return None
    import json

    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def merge_operating_point(fusion: dict[str, Any], op: dict[str, Any] | None) -> dict[str, Any]:
    mix = dict(fusion or {})
    if not op:
        return mix
    status = str(op.get("status") or "")
    if status not in {"provisional", "locked"}:
        return mix
    for key, src in (("t_lsb", "t_lsb"), ("t_freq", "t_freq"), ("t_class", "t_class")):
        if src in op and op[src] is not None:
            mix[key] = float(op[src])
    if op.get("id"):
        mix["operating_point_id"] = str(op["id"])
    if op.get("fusion_mode"):
        mix["mode"] = str(op["fusion_mode"])
    return mix


def resolve_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
