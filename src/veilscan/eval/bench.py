"""Frozen generator bench for an operating point. Not a camera corpus."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from veilscan.config import DEFAULT_PEAK_OK, VeilConfig
from veilscan.engine import analyze_image
from veilscan.eval.metrics import auc_roc, cut_at_fpr, f1_at, fpr_at, tpr_at, tpr_at_fpr
from veilscan.eval.robustness import apply_attack
from veilscan.generators import embed, synthetic_cover
from veilscan.registry import all_detectors, ensure_loaded

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
) -> dict[str, Any]:
    proto = dict(protocol or load_protocol())
    n = int(n if n is not None else proto.get("n", 20))
    size = int(size if size is not None else proto.get("size", 128))
    seed = int(proto.get("seed", 20260822))
    hold = int(proto.get("holdout_seed_start", 10000))
    styles = list(styles or proto.get("styles") or ["photo"])
    families = list(families or proto.get("families") or ["lsb", "dct"])
    attacks = list(attacks or proto.get("attacks") or ["identity"])
    cfg = cfg or VeilConfig.load()
    cfg.apply_calibration = False
    names = detectors or _detector_names(proto, cfg)
    rng = np.random.default_rng(seed)

    cells: dict[str, Any] = {}
    cover_bucket: dict[tuple[str, str], dict[str, list[float]]] = {}

    for style in styles:
        for attack in attacks:
            cover_scores: list[float] = []
            cover_lsb: list[float] = []
            cover_freq: list[float] = []
            cover_class: list[float] = []
            by_fam: dict[str, dict[str, list]] = {}
            for i in range(n):
                cover_seed = hold + i + (0 if style == "sine" else 50_000)
                cover = synthetic_cover(size, size, np.random.default_rng(cover_seed), style=style)
                atk_rng = np.random.default_rng(int(rng.integers(1 << 30)))
                cover_a = apply_attack(cover, attack, atk_rng)
                c_res = analyze_image(cover_a, cfg, names)
                cover_scores.append(float(c_res.score))
                cover_lsb.append(float(c_res.lsb_score))
                cover_freq.append(float(c_res.freq_score))
                cover_class.append(float(c_res.class_score))
                for fam in families:
                    fam_off = 17 + 97 * families.index(fam)
                    marked = embed(cover, fam, seed=cover_seed + fam_off)
                    marked_a = apply_attack(marked, attack, np.random.default_rng(cover_seed + 91))
                    m_res = analyze_image(marked_a, cfg, names)
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
        "corpus": "generator",
        "cells": cells,
        "notes": proto.get("notes"),
    }
    report["operating_point"] = choose_operating_point(report, cover_bucket, proto)
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
    return {
        "id": f"op-v0.4.0-{status}-n{n}",
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
        "corpus": "generator-photo" if "photo" in styles else "generator",
        "slice": [f"{s}/{a}" for s in styles for a in attacks],
        "fusion_mode": "legacy",
        "notes": "Generator covers, not camera photos. Do not cite as ImageNet FPR.",
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
            "",
        ]
    )
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
        op_path.write_text(
            json.dumps(_clean_nans(report["operating_point"]), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        written["operating_point"] = str(op_path)
    return written
