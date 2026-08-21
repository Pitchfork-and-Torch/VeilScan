"""Synthetic eval harness used by `veilscan selftest` and `veilscan eval`."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veilscan.config import VeilConfig
from veilscan.engine import analyze_image
from veilscan.eval.metrics import auc_roc, f1_at, tpr_at_fpr
from veilscan.eval.robustness import DEFAULT_ATTACKS, apply_attack
from veilscan.generators import FAMILIES, embed, synthetic_cover
from veilscan.image_io import load_rgb


def run_synthetic(
    n: int = 8,
    size: int = 128,
    families: list[str] | None = None,
    seed: int = 0,
    cfg: VeilConfig | None = None,
    detectors: list[str] | None = None,
) -> dict:
    cfg = cfg or VeilConfig.load()
    # Selftest should stay fast: skip WMD (needs refs) and untrained deep nets
    # still skip themselves.
    families = families or list(FAMILIES)
    rng = np.random.default_rng(seed)
    per_family = {}
    all_y = []
    all_s = []
    for fam in families:
        y = []
        s = []
        for i in range(n):
            cover = synthetic_cover(size, size, np.random.default_rng(int(rng.integers(1 << 30))))
            marked = embed(cover, fam, seed=int(rng.integers(1 << 30)))
            c_res = analyze_image(cover, cfg, detectors)
            m_res = analyze_image(marked, cfg, detectors)
            y.extend([0, 1])
            s.extend([c_res.score, m_res.score])
        y_a = np.asarray(y)
        s_a = np.asarray(s)
        per_family[fam] = {
            "auc": auc_roc(y_a, s_a),
            "tpr_at_fpr_5": tpr_at_fpr(y_a, s_a, 0.05),
            "f1": f1_at(y_a, s_a, cfg.threshold),
            "mean_cover": float(s_a[y_a == 0].mean()),
            "mean_marked": float(s_a[y_a == 1].mean()),
        }
        all_y.append(y_a)
        all_s.append(s_a)
    y_all = np.concatenate(all_y)
    s_all = np.concatenate(all_s)
    return {
        "overall": {
            "auc": auc_roc(y_all, s_all),
            "tpr_at_fpr_5": tpr_at_fpr(y_all, s_all, 0.05),
            "tpr_at_fpr_1": tpr_at_fpr(y_all, s_all, 0.01),
            "f1": f1_at(y_all, s_all, cfg.threshold),
        },
        "families": per_family,
        "n": n,
        "size": size,
    }


def run_folder_eval(clean_dir: str | Path, wm_dir: str | Path, cfg: VeilConfig | None = None) -> dict:
    from veilscan.image_io import iter_images

    cfg = cfg or VeilConfig.load()
    y = []
    s = []
    for p in iter_images(clean_dir):
        r = analyze_image(load_rgb(p), cfg)
        y.append(0)
        s.append(r.score)
    for p in iter_images(wm_dir):
        r = analyze_image(load_rgb(p), cfg)
        y.append(1)
        s.append(r.score)
    y_a = np.asarray(y)
    s_a = np.asarray(s)
    return {
        "auc": auc_roc(y_a, s_a),
        "tpr_at_fpr_5": tpr_at_fpr(y_a, s_a, 0.05),
        "tpr_at_fpr_1": tpr_at_fpr(y_a, s_a, 0.01),
        "f1": f1_at(y_a, s_a, cfg.threshold),
        "n_clean": int((y_a == 0).sum()),
        "n_wm": int((y_a == 1).sum()),
    }


def run_robustness(n: int = 4, size: int = 128, family: str = "dct", seed: int = 1) -> dict:
    cfg = VeilConfig.load()
    rng = np.random.default_rng(seed)
    out = {}
    for attack in DEFAULT_ATTACKS:
        y = []
        s = []
        for i in range(n):
            cover = synthetic_cover(size, size, np.random.default_rng(int(rng.integers(1 << 30))))
            marked = embed(cover, family, seed=int(rng.integers(1 << 30)))
            cover_a = apply_attack(cover, attack, rng)
            marked_a = apply_attack(marked, attack, rng)
            y.extend([0, 1])
            s.extend([analyze_image(cover_a, cfg).score, analyze_image(marked_a, cfg).score])
        y_a, s_a = np.asarray(y), np.asarray(s)
        from veilscan.eval.metrics import auc_roc

        out[attack] = {"auc": auc_roc(y_a, s_a)}
    return {"family": family, "attacks": out}
