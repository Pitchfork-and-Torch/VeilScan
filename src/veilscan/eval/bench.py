"""Frozen generator bench for an operating point. Not a camera corpus."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from veilscan.config import DEFAULT_PEAK_OK, VeilConfig, load_camera_operating_point, load_operating_point
from veilscan.engine import analyze_image
from veilscan.eval.metrics import auc_roc, cut_at_fpr, f1_at, fpr_at, tpr_at, tpr_at_fpr
from veilscan.eval.robustness import apply_attack
from veilscan.generators import embed, synthetic_cover
from veilscan.image_io import load_rgb
from veilscan.registry import all_detectors, ensure_loaded

_COVER_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_cover_paths(root: str | Path) -> list[Path]:
    base = Path(root)
    if not base.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _COVER_EXT:
            continue
        low = str(p).replace("\\", "/").lower()
        if "screenshot" in low or "screen-shot" in low:
            continue
        out.append(p)
    return out


def load_camera_cover_pack(
    root: str | Path, n: int, size: int, seed: int = 20260822
) -> tuple[list[np.ndarray], list[bool]]:
    import cv2

    paths = iter_cover_paths(root)
    if not paths:
        raise FileNotFoundError(f"no camera images under {root}")
    rng = np.random.default_rng(seed)
    if len(paths) > n:
        pick = rng.choice(len(paths), size=n, replace=False)
        paths = [paths[int(i)] for i in sorted(pick.tolist())]
    else:
        paths = paths[:n]
    covers: list[np.ndarray] = []
    jpeg_flags: list[bool] = []
    for p in paths:
        rgb = load_rgb(p)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            continue
        covers.append(cv2.resize(rgb[..., :3], (size, size), interpolation=cv2.INTER_AREA))
        jpeg_flags.append(p.suffix.lower() in {".jpg", ".jpeg"})
    if not covers:
        raise FileNotFoundError(f"no readable RGB images under {root}")
    return covers, jpeg_flags


def load_camera_covers(root: str | Path, n: int, size: int, seed: int = 20260822) -> list[np.ndarray]:
    rgb, _ = load_camera_cover_pack(root, n, size, seed=seed)
    return rgb

def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if not np.isfinite(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def _clean_nans(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "configs" / "bench_protocol.yaml"
DEFAULT_JSON = ROOT / "docs" / "bench" / "latest.json"
DEFAULT_MD = ROOT / "docs" / "bench" / "latest.md"
DEFAULT_OP = ROOT / "configs" / "operating_point.json"


def load_protocol(path: str | Path | None = None) -> dict[str, Any]:
    src = Path(path) if path else DEFAULT_PROTOCOL
    with src.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data["_path"] = str(src)
    return data


def _detector_names(protocol: dict[str, Any], cfg: VeilConfig) -> list[str]:
    spec = protocol.get("detectors") or "peak_ok"
    if spec == "peak_ok":
        return list(cfg.peak_ok or DEFAULT_PEAK_OK)
    if spec == "all":
        ensure_loaded()
        return [d.name for d in all_detectors() if d.name not in {"wmd"}]
    return list(spec)


def _cell_stats(y: list[int], s: list[float], threshold: float) -> dict[str, float]:
    y_a = np.asarray(y)
    s_a = np.asarray(s, dtype=np.float64)
    return {
        "auc": auc_roc(y_a, s_a),
        "tpr_at_fpr_5": tpr_at_fpr(y_a, s_a, 0.05),
        "tpr_at_fpr_1": tpr_at_fpr(y_a, s_a, 0.01),
        "f1": f1_at(y_a, s_a, threshold),
        "fpr_at_threshold": fpr_at(y_a, s_a, threshold),
        "tpr_at_threshold": tpr_at(y_a, s_a, threshold),
        "mean_cover": float(s_a[y_a == 0].mean()) if np.any(y_a == 0) else float("nan"),
        "mean_marked": float(s_a[y_a == 1].mean()) if np.any(y_a == 1) else float("nan"),
        "n_cover": int((y_a == 0).sum()),
        "n_marked": int((y_a == 1).sum()),
    }


def run_bench(
    protocol: dict[str, Any] | None = None,
    *,
    n: int | None = None,
    size: int | None = None,
    styles: list[str] | None = None,
    families: list[str] | None = None,
    attacks: list[str] | None = None,
    cfg: VeilConfig | None = None,
    detectors: list[str] | None = None,
    covers: list[np.ndarray] | None = None,
    corpus_id: str | None = None,
    jpeg_container: list[bool] | None = None,
) -> dict[str, Any]:
    proto = dict(protocol or load_protocol())
    n = int(n if n is not None else proto.get("n", 20))
    size = int(size if size is not None else proto.get("size", 128))
    seed = int(proto.get("seed", 20260822))
    hold = int(proto.get("holdout_seed_start", 10000))
    families = list(families or proto.get("families") or ["lsb", "dct"])
    attacks = list(attacks or proto.get("attacks") or ["identity"])
    camera = covers is not None and len(covers) > 0
    if camera:
        covers = [np.asarray(c) for c in covers]
        n = min(n, len(covers))
        covers = covers[:n]
        styles = ["camera"]
        proto = dict(proto)
        lock_attacks = [a for a in attacks if a in {"identity", "jpeg_70"}] or list(attacks)
        proto["operating_slice"] = {"styles": ["camera"], "attacks": lock_attacks}
    else:
        styles = list(styles or proto.get("styles") or ["photo"])
    cfg = cfg or VeilConfig.load()
    cfg.apply_calibration = False
    names = detectors or _detector_names(proto, cfg)
    rng = np.random.default_rng(seed)

    cells: dict[str, Any] = {}
    cover_bucket: dict[tuple[str, str], dict[str, list[float]]] = {}
    marked_bucket: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}

    for style in styles:
        for attack in attacks:
            cover_scores: list[float] = []
            cover_lsb: list[float] = []
            cover_freq: list[float] = []
            cover_class: list[float] = []
            by_fam: dict[str, dict[str, list]] = {}
            for i in range(n):
                cover_seed = hold + i + (0 if style == "sine" else 50_000)
                if camera:
                    cover = covers[i]
                    if cover.shape[0] != size or cover.shape[1] != size:
                        import cv2

                        cover = cv2.resize(cover, (size, size), interpolation=cv2.INTER_AREA)
                else:
                    cover = synthetic_cover(size, size, np.random.default_rng(cover_seed), style=style)
                atk_rng = np.random.default_rng(int(rng.integers(1 << 30)))
                cover_a = apply_attack(cover, attack, atk_rng)
                src_jpeg = bool(jpeg_container[i]) if jpeg_container is not None and i < len(jpeg_container) else False
                hint = src_jpeg or attack.startswith("jpeg")
                q_est = None
                if attack.startswith("jpeg_") and attack.split("_")[-1].isdigit():
                    q_est = int(attack.split("_")[-1])
                c_res = analyze_image(cover_a, cfg, names, jpeg_container=hint, jpeg_quality_est=q_est)
                cover_scores.append(float(c_res.score))
                cover_lsb.append(float(c_res.lsb_score))
                cover_freq.append(float(c_res.freq_score))
                cover_class.append(float(c_res.class_score))
                for fam in families:
                    fam_off = 17 + 97 * families.index(fam)
                    marked = embed(cover, fam, seed=cover_seed + fam_off)
                    marked_a = apply_attack(marked, attack, np.random.default_rng(cover_seed + 91))
                    m_res = analyze_image(marked_a, cfg, names, jpeg_container=hint, jpeg_quality_est=q_est)
                    rec = by_fam.setdefault(fam, {"y": [], "score": [], "lsb": [], "freq": [], "klass": []})
                    rec["y"].extend([0, 1])
                    rec["score"].extend([float(c_res.score), float(m_res.score)])
                    rec["lsb"].extend([float(c_res.lsb_score), float(m_res.lsb_score)])
                    rec["freq"].extend([float(c_res.freq_score), float(m_res.freq_score)])
                    rec["klass"].extend([float(c_res.class_score), float(m_res.class_score)])
            cover_bucket[(style, attack)] = {
                "ensemble": cover_scores,
                "lsb": cover_lsb,
                "freq": cover_freq,
                "klass": cover_class,
            }
            marked_bucket[(style, attack)] = {
                fam: {
                    "ensemble": rec["score"][1::2],
                    "lsb": rec["lsb"][1::2],
                    "freq": rec["freq"][1::2],
                    "klass": rec["klass"][1::2],
                }
                for fam, rec in by_fam.items()
            }
            fam_metrics = {}
            for fam, rec in by_fam.items():
                fam_metrics[fam] = {
                    "ensemble": _cell_stats(rec["y"], rec["score"], cfg.threshold),
                    "lsb_head": _cell_stats(rec["y"], rec["lsb"], float(cfg.fusion.get("t_lsb", 0.5))),
                    "freq_head": _cell_stats(rec["y"], rec["freq"], float(cfg.fusion.get("t_freq", 0.5))),
                    "class_head": _cell_stats(rec["y"], rec["klass"], float(cfg.fusion.get("t_class", cfg.threshold))),
                }
            key = f"{style}/{attack}"
            cells[key] = {
                "families": fam_metrics,
                "cover": {
                    "ensemble_mean": float(np.mean(cover_scores)),
                    "n": n,
                },
            }

    report = {
        "protocol_id": proto.get("id"),
        "protocol_path": proto.get("_path"),
        "n": n,
        "size": size,
        "seed": seed,
        "styles": styles,
        "families": families,
        "attacks": attacks,
        "detectors": names,
        "threshold_used": cfg.threshold,
        "corpus": "camera" if camera else "generator",
        "corpus_id": (corpus_id or ("camera" if camera else "generator")),
        "cells": cells,
        "notes": proto.get("notes"),
    }
    report["operating_point"] = choose_operating_point(report, cover_bucket, proto)
    report["fpr_at_locks"] = fpr_at_locks(cover_bucket, proto)
    report["ab"] = ab_fusion(cover_bucket, marked_bucket, report["operating_point"], proto)
    if report["ab"].get("flip_default"):
        report["operating_point"]["fusion_mode"] = "specialist_or"
        report["operating_point"]["notes"] = (
            str(report["operating_point"].get("notes") or "")
            + " specialist_or won A/B on this slice (FPR did not rise)."
        ).strip()
    else:
        report["operating_point"]["fusion_mode"] = "legacy"
        report["operating_point"]["notes"] = (
            str(report["operating_point"].get("notes") or "")
            + " specialist_or did not beat legacy FPR; default stays legacy."
        ).strip()
    return report


def choose_operating_point(
    report: dict[str, Any],
    cover_bucket: dict[tuple[str, str], dict[str, list[float]]],
    proto: dict[str, Any],
) -> dict[str, Any]:
    fpr_target = float(proto.get("fpr_target", 0.05))
    slice_cfg = proto.get("operating_slice") or {}
    styles = list(slice_cfg.get("styles") or ["photo"])
    attacks = list(slice_cfg.get("attacks") or ["identity"])
    ens: list[float] = []
    lsb: list[float] = []
    freq: list[float] = []
    klass: list[float] = []
    for style in styles:
        for attack in attacks:
            bucket = cover_bucket.get((style, attack))
            if not bucket:
                continue
            ens.extend(bucket["ensemble"])
            lsb.extend(bucket["lsb"])
            freq.extend(bucket["freq"])
            klass.extend(bucket["klass"])
    t_ens = cut_at_fpr(np.asarray(ens, dtype=np.float64), fpr_target) if ens else 0.48
    t_lsb = cut_at_fpr(np.asarray(lsb, dtype=np.float64), fpr_target) if lsb else 0.50
    t_freq = cut_at_fpr(np.asarray(freq, dtype=np.float64), fpr_target) if freq else 0.50
    t_class = cut_at_fpr(np.asarray(klass, dtype=np.float64), fpr_target) if klass else 0.48
    fpr_est = fpr_at(np.zeros(len(ens), dtype=np.int32), np.asarray(ens), t_ens) if ens else None
    n = report.get("n") or 0
    status = "provisional" if int(n) < 50 else "locked"
    camera = styles == ["camera"] or report.get("corpus") == "camera"
    if camera:
        from veilscan import __version__

        slug = str(report.get("corpus_id") or "camera")
        oid = f"op-v{__version__}-{slug}-{status}-n{n}"
        notes = "Camera stills pack sidecar. Does not replace the generator lock. Do not cite as UniFreq/ImageNet FPR."
        corpus = slug
    else:
        oid = f"op-v0.4.0-{status}-n{n}"
        notes = "Generator covers, not camera photos. Do not cite as ImageNet FPR."
        corpus = "generator-photo" if "photo" in styles else "generator"
    return {
        "id": oid,
        "status": status,
        "threshold": round(float(t_ens), 6),
        "t_lsb": round(float(t_lsb), 6),
        "t_freq": round(float(t_freq), 6),
        "t_class": round(float(t_class), 6),
        "fpr_est": None if fpr_est is None or not np.isfinite(fpr_est) else round(float(fpr_est), 6),
        "fpr_target": fpr_target,
        "n": n,
        "date": date.today().isoformat(),
        "protocol": report.get("protocol_id"),
        "corpus": corpus,
        "slice": [f"{s}/{a}" for s in styles for a in attacks],
        "fusion_mode": "legacy",
        "notes": notes,
    }


def _concat_cover_ensemble(
    cover_bucket: dict[tuple[str, str], dict[str, list[float]]],
    proto: dict[str, Any],
) -> np.ndarray:
    slice_cfg = proto.get("operating_slice") or {}
    styles = list(slice_cfg.get("styles") or ["photo"])
    attacks = list(slice_cfg.get("attacks") or ["identity"])
    ens: list[float] = []
    for style in styles:
        for attack in attacks:
            bucket = cover_bucket.get((style, attack))
            if not bucket:
                continue
            ens.extend(bucket["ensemble"])
    return np.asarray(ens, dtype=np.float64)


def fpr_at_named_thresholds(scores: np.ndarray, locks: dict[str, float]) -> dict[str, float]:
    """FPR treating every score as a cover. Used to test existing locks on a new corpus."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.zeros(s.size, dtype=np.int32)
    out: dict[str, float] = {}
    for name, thr in locks.items():
        val = fpr_at(y, s, float(thr))
        out[str(name)] = float("nan") if val is None or not np.isfinite(val) else round(float(val), 6)
    return out


def fpr_at_locks(
    cover_bucket: dict[tuple[str, str], dict[str, list[float]]],
    proto: dict[str, Any],
) -> dict[str, Any]:
    scores = _concat_cover_ensemble(cover_bucket, proto)
    rows: dict[str, Any] = {
        "n": int(scores.size),
        "cover_mean": None if scores.size == 0 else round(float(scores.mean()), 6),
        "locks": {},
    }
    named: dict[str, dict[str, Any]] = {}
    gen = load_operating_point()
    cam = load_camera_operating_point()
    for key, op in (("generator", gen), ("camera_bsds", cam)):
        if not op or op.get("threshold") is None:
            continue
        thr = float(op["threshold"])
        fpr = fpr_at_named_thresholds(scores, {key: thr}).get(key)
        named[key] = {
            "id": op.get("id"),
            "threshold": thr,
            "fpr": fpr,
        }
    rows["locks"] = named
    return rows


def _or_hits(lsb: np.ndarray, freq: np.ndarray, klass: np.ndarray, op: dict[str, Any]) -> np.ndarray:
    t_lsb = float(op.get("t_lsb") or 0.5)
    t_freq = float(op.get("t_freq") or 0.5)
    t_class = float(op.get("t_class") or 0.48)
    return (lsb >= t_lsb) | (freq >= t_freq) | (klass >= t_class)


def ab_fusion(
    cover_bucket: dict[tuple[str, str], dict[str, list[float]]],
    marked_bucket: dict[tuple[str, str], dict[str, dict[str, list[float]]]],
    op: dict[str, Any],
    proto: dict[str, Any],
) -> dict[str, Any]:
    """Same-slice FPR/TPR for legacy threshold vs specialist-OR. Not a nested holdout."""
    slice_cfg = proto.get("operating_slice") or {}
    styles = list(slice_cfg.get("styles") or ["photo"])
    attacks = list(slice_cfg.get("attacks") or ["identity"])
    thr = float(op.get("threshold") or 0.48)
    ens_c: list[float] = []
    lsb_c: list[float] = []
    freq_c: list[float] = []
    klass_c: list[float] = []
    for style in styles:
        for attack in attacks:
            bucket = cover_bucket.get((style, attack))
            if not bucket:
                continue
            ens_c.extend(bucket["ensemble"])
            lsb_c.extend(bucket["lsb"])
            freq_c.extend(bucket["freq"])
            klass_c.extend(bucket["klass"])
    ens_a = np.asarray(ens_c, dtype=np.float64)
    zeros = np.zeros(ens_a.size, dtype=np.int32)
    legacy_fpr = fpr_at(zeros, ens_a, thr) if ens_a.size else None
    or_hits = _or_hits(
        np.asarray(lsb_c, dtype=np.float64),
        np.asarray(freq_c, dtype=np.float64),
        np.asarray(klass_c, dtype=np.float64),
        op,
    )
    or_fpr = float(or_hits.mean()) if or_hits.size else None

    def tpr_pair(fam: str) -> tuple[float | None, float | None]:
        ens_m: list[float] = []
        lsb_m: list[float] = []
        freq_m: list[float] = []
        klass_m: list[float] = []
        for style in styles:
            for attack in attacks:
                rec = (marked_bucket.get((style, attack)) or {}).get(fam)
                if not rec:
                    continue
                ens_m.extend(rec["ensemble"])
                lsb_m.extend(rec["lsb"])
                freq_m.extend(rec["freq"])
                klass_m.extend(rec["klass"])
        if not ens_m:
            return None, None
        ones = np.ones(len(ens_m), dtype=np.int32)
        leg = tpr_at(ones, np.asarray(ens_m, dtype=np.float64), thr)
        orr = float(
            _or_hits(
                np.asarray(lsb_m, dtype=np.float64),
                np.asarray(freq_m, dtype=np.float64),
                np.asarray(klass_m, dtype=np.float64),
                op,
            ).mean()
        )
        return leg, orr

    lsb_leg, lsb_or = tpr_pair("lsb")
    dct_leg, dct_or = tpr_pair("dct")
    fpr_ok = or_fpr is not None and legacy_fpr is not None and or_fpr <= legacy_fpr + 0.005
    tpr_ok = True
    if lsb_leg is not None and lsb_or is not None:
        tpr_ok = tpr_ok and lsb_or + 1e-9 >= lsb_leg - 0.02
    if dct_leg is not None and dct_or is not None:
        tpr_ok = tpr_ok and dct_or + 1e-9 >= dct_leg - 0.02
    flip = bool(fpr_ok and tpr_ok)
    nested = nested_holdout_fpr(
        np.asarray(ens_c, dtype=np.float64),
        np.asarray(lsb_c, dtype=np.float64),
        np.asarray(freq_c, dtype=np.float64),
        np.asarray(klass_c, dtype=np.float64),
        float(proto.get("fpr_target") or 0.05),
    )
    return {
        "legacy_fpr": None if legacy_fpr is None else round(float(legacy_fpr), 6),
        "or_fpr": None if or_fpr is None else round(float(or_fpr), 6),
        "legacy_tpr_lsb": None if lsb_leg is None else round(float(lsb_leg), 6),
        "or_tpr_lsb": None if lsb_or is None else round(float(lsb_or), 6),
        "legacy_tpr_dct": None if dct_leg is None else round(float(dct_leg), 6),
        "or_tpr_dct": None if dct_or is None else round(float(dct_or), 6),
        "flip_default": flip,
        "nested_holdout": nested,
        "notes": "In-sample A/B plus even/odd nested FPR. Do not flip specialist-OR without nested FPR holding.",
    }


def nested_holdout_fpr(
    ens: np.ndarray,
    lsb: np.ndarray,
    freq: np.ndarray,
    klass: np.ndarray,
    fpr_target: float,
) -> dict[str, Any] | None:
    n = int(ens.size)
    if n < 8:
        return None
    fit_idx = np.arange(0, n, 2)
    ev_idx = np.arange(1, n, 2)
    op_fit = {
        "threshold": cut_at_fpr(ens[fit_idx], fpr_target),
        "t_lsb": cut_at_fpr(lsb[fit_idx], fpr_target),
        "t_freq": cut_at_fpr(freq[fit_idx], fpr_target),
        "t_class": cut_at_fpr(klass[fit_idx], fpr_target),
    }
    zeros = np.zeros(ev_idx.size, dtype=np.int32)
    legacy = fpr_at(zeros, ens[ev_idx], float(op_fit["threshold"]))
    or_hits = _or_hits(lsb[ev_idx], freq[ev_idx], klass[ev_idx], op_fit)
    or_fpr = float(or_hits.mean()) if or_hits.size else None
    return {
        "n_fit": int(fit_idx.size),
        "n_eval": int(ev_idx.size),
        "legacy_fpr": None if legacy is None or not np.isfinite(legacy) else round(float(legacy), 6),
        "or_fpr": None if or_fpr is None else round(float(or_fpr), 6),
        "flip_default": bool(
            or_fpr is not None and legacy is not None and np.isfinite(legacy) and or_fpr <= float(legacy) + 0.005
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# VeilScan bench",
        "",
        f"protocol `{report.get('protocol_id')}` n={report.get('n')} size={report.get('size')} seed={report.get('seed')}",
        "",
        report.get("notes") or "",
        "",
        "| slice | family | AUC | TPR@5%FPR | mean cover | mean marked |",
        "|-------|--------|-----|-----------|------------|-------------|",
    ]
    for slice_key, cell in (report.get("cells") or {}).items():
        for fam, metrics in (cell.get("families") or {}).items():
            ens = metrics.get("ensemble") or {}
            lines.append(
                f"| {slice_key} | {fam} | {ens.get('auc', float('nan')):.3f} | "
                f"{ens.get('tpr_at_fpr_5', float('nan')):.3f} | "
                f"{ens.get('mean_cover', float('nan')):.3f} | "
                f"{ens.get('mean_marked', float('nan')):.3f} |"
            )
    op = report.get("operating_point") or {}
    lines.extend(
        [
            "",
            "## Operating point (derived)",
            "",
            f"- id: `{op.get('id')}` status `{op.get('status')}`",
            f"- threshold={op.get('threshold')} t_lsb={op.get('t_lsb')} t_freq={op.get('t_freq')} t_class={op.get('t_class')}",
            f"- fpr_est={op.get('fpr_est')} n={op.get('n')}",
            f"- fusion_mode={op.get('fusion_mode')}",
            "",
        ]
    )
    ab = report.get("ab") or {}
    if ab:
        lines.extend(
            [
                "## A/B legacy vs specialist-OR",
                "",
                f"- legacy_fpr={ab.get('legacy_fpr')} or_fpr={ab.get('or_fpr')}",
                f"- LSB TPR legacy={ab.get('legacy_tpr_lsb')} or={ab.get('or_tpr_lsb')}",
                f"- DCT TPR legacy={ab.get('legacy_tpr_dct')} or={ab.get('or_tpr_dct')}",
                f"- flip_default={ab.get('flip_default')}",
                "",
            ]
        )
        nested = ab.get("nested_holdout") or {}
        if nested:
            lines.extend(
                [
                    f"- nested n_fit={nested.get('n_fit')} n_eval={nested.get('n_eval')} "
                    f"legacy_fpr={nested.get('legacy_fpr')} or_fpr={nested.get('or_fpr')} "
                    f"flip={nested.get('flip_default')}",
                    "",
                ]
            )
    locks = report.get("fpr_at_locks") or {}
    named = locks.get("locks") or {}
    if named:
        lines.extend(["## FPR at existing locks", ""])
        lines.append(f"- n={locks.get('n')} cover_mean={locks.get('cover_mean')}")
        for key, row in named.items():
            lines.append(
                f"- {key}: id=`{row.get('id')}` threshold={row.get('threshold')} fpr={row.get('fpr')}"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_outputs(
    report: dict[str, Any],
    json_path: Path | None = None,
    md_path: Path | None = None,
    operating_point_path: Path | None = None,
    write_operating_point: bool = False,
) -> dict[str, str]:
    json_path = json_path or DEFAULT_JSON
    md_path = md_path or DEFAULT_MD
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _clean_nans(report)
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    written = {"json": str(json_path), "md": str(md_path)}
    if write_operating_point:
        op_path = operating_point_path or DEFAULT_OP
        corpus = str(report.get("corpus") or "")
        if corpus not in {"generator", "generator-photo"} and Path(op_path).resolve() == DEFAULT_OP.resolve():
            raise ValueError("refusing to overwrite generator operating_point.json from a camera bench")
        op_path.write_text(
            json.dumps(_clean_nans(report["operating_point"]), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        written["operating_point"] = str(op_path)
    return written
