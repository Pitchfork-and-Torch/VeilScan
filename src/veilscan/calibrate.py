"""Affine-logit calibration. Identity maps are a no-op."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = 1


def _pkg_cal_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "calibration.json"


def identity_calibration() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "method": "affine_logit",
        "detectors": {},
        "ensemble": {"a": 1.0, "b": 0.0},
        "fit": {"n": 0, "protocol": "identity"},
    }


def load_calibration(path: str | Path | None = None) -> dict[str, Any]:
    src = Path(path) if path else _pkg_cal_path()
    if not src.is_file():
        return identity_calibration()
    with src.open("r", encoding="utf-8") as f:
        data = json.load(f)
    base = identity_calibration()
    base.update(data or {})
    base.setdefault("detectors", {})
    base.setdefault("ensemble", {"a": 1.0, "b": 0.0})
    return base


def logit(p: float) -> float:
    x = float(np.clip(p, 1e-4, 1.0 - 1e-4))
    return float(np.log(x / (1.0 - x)))


def sigmoid(z: float) -> float:
    z = float(np.clip(z, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def apply_affine(params: dict[str, float] | None, score: float) -> float:
    if not params:
        return float(np.clip(score, 0.0, 1.0))
    a = float(params.get("a", 1.0))
    b = float(params.get("b", 0.0))
    return float(np.clip(sigmoid(a * logit(score) + b), 0.0, 1.0))


def _fit_one(scores: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Least-squares a,b on logit(score) -> logit(clipped y-smoothed). Unstable y uses identity."""
    if scores.size < 8 or y.min() == y.max() or scores.std() < 1e-6:
        return {"a": 1.0, "b": 0.0}
    x = np.array([logit(s) for s in scores], dtype=np.float64)
    # Smooth labels so logit is defined; map 0->0.05, 1->0.95 then logit.
    t = 0.05 + 0.90 * y.astype(np.float64)
    z = np.array([logit(v) for v in t], dtype=np.float64)
    A = np.vstack([x, np.ones(x.size)]).T
    try:
        a, b = np.linalg.lstsq(A, z, rcond=None)[0]
    except np.linalg.LinAlgError:
        return {"a": 1.0, "b": 0.0}
    a = float(np.clip(a, 0.15, 6.0))
    b = float(np.clip(b, -4.0, 4.0))
    return {"a": a, "b": b}


def fit_calibration(
    n: int = 6,
    size: int = 96,
    families: list[str] | None = None,
    seed: int = 0,
    detectors: list[str] | None = None,
) -> dict[str, Any]:
    """Fit per-head and ensemble maps on synthetic pairs. Classical path; no GPU."""
    from veilscan.config import VeilConfig
    from veilscan.engine import analyze_image
    from veilscan.generators import FAMILIES, embed, synthetic_cover

    cfg = VeilConfig.load()
    cfg.apply_calibration = False
    families = families or ["lsb", "dct", "dwt", "spread"]
    families = [f for f in families if f in FAMILIES]
    rng = np.random.default_rng(seed)
    ens_s: list[float] = []
    ens_y: list[int] = []
    per: dict[str, list[float]] = {}
    per_y: dict[str, list[int]] = {}
    for fam in families:
        for _ in range(n):
            cover = synthetic_cover(size, size, np.random.default_rng(int(rng.integers(1 << 30))), style="mix")
            marked = embed(cover, fam, seed=int(rng.integers(1 << 30)))
            for img, lab in ((cover, 0), (marked, 1)):
                r = analyze_image(img, cfg, detectors)
                ens_s.append(r.score)
                ens_y.append(lab)
                for d in r.detectors:
                    if d.skipped:
                        continue
                    per.setdefault(d.detector, []).append(d.score)
                    per_y.setdefault(d.detector, []).append(lab)
    det_maps = {}
    for name, ss in per.items():
        det_maps[name] = _fit_one(np.asarray(ss), np.asarray(per_y[name]))
    out = {
        "schema_version": SCHEMA,
        "method": "affine_logit",
        "detectors": det_maps,
        "ensemble": _fit_one(np.asarray(ens_s), np.asarray(ens_y)),
        "fit": {"n": n, "size": size, "families": families, "protocol": "synthetic_mix"},
    }
    return out


def save_calibration(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
