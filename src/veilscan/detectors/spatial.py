"""Classical spatial / statistical detectors."""

from __future__ import annotations

import numpy as np
from scipy import stats

from veilscan.detectors.base import BaseDetector
from veilscan.dsp import entropy, score_from_stat, to_gray
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult


def _u8(channel: np.ndarray) -> np.ndarray:
    x = np.asarray(channel)
    if x.dtype != np.uint8:
        x = np.clip(np.rint(x), 0, 255).astype(np.uint8)
    return x


class ChiSquareDetector(BaseDetector):
    name = "chi_square"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = _u8(to_gray(image))
        extras = {}
        rgb_p = []
        for label, ch in [("r", image[..., 0]), ("g", image[..., 1]), ("b", image[..., 2])]:
            p, chi, dof = _chi_square_pairs(_u8(ch))
            rgb_p.append(p)
            extras[f"{label}_p"] = p
            extras[f"{label}_chi"] = chi
            extras[f"{label}_dof"] = dof
        p_gray, chi_g, dof_g = _chi_square_pairs(gray)
        extras["gray_p"] = p_gray
        extras["gray_chi"] = chi_g
        extras["gray_dof"] = dof_g
        # Mean RGB p-value. Gray is a mix and can look random after interpolation.
        p = float(np.mean(rgb_p)) if rgb_p else float(p_gray)
        score = float(p)
        conf = 0.85 if extras["r_dof"] >= 20 else 0.45
        expl = (
            f"Chi-square LSB pair test (Westfeld-Pfitzmann). "
            f"Max channel p-value={p:.4f} (high => equalized even/odd bins, typical of LSB)."
        )
        return DetectionResult(self.name, score, conf, expl, extras=extras, tier=self.tier).clamp()


def _chi_square_pairs(channel: np.ndarray) -> tuple[float, float, int]:
    hist = np.bincount(channel.ravel(), minlength=256).astype(np.float64)[:256]
    chi = 0.0
    dof = 0
    for k in range(128):
        n1 = hist[2 * k]
        n2 = hist[2 * k + 1]
        exp = 0.5 * (n1 + n2)
        if exp >= 5.0:
            chi += ((n1 - exp) ** 2 + (n2 - exp) ** 2) / exp
            dof += 1
    if dof < 2:
        return 0.0, chi, dof
    p = float(stats.chi2.sf(chi, dof))
    return p, float(chi), int(dof)


class RSAnalysisDetector(BaseDetector):
    name = "rs_analysis"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = _u8(to_gray(image))
        stats_m = _rs_mask(gray, np.array([0, 1, 1, 0], dtype=np.int8))
        extras = {**stats_m}
        d = abs(stats_m["rm"] - stats_m["sm"] - (stats_m["rneg"] - stats_m["sneg"]))
        # Clean natural images usually keep d small; LSB mixing inflates it.
        score = score_from_stat(d, center=0.04, scale=0.06)
        p_hat = _rs_payload_estimate(stats_m)
        extras["d"] = d
        extras["payload_hat"] = p_hat
        score = max(score, score_from_stat(p_hat, center=0.08, scale=0.12))
        expl = (
            f"RS analysis (Fridrich). Regular/singular gap d={d:.4f}, "
            f"rough payload hat={p_hat:.3f}."
        )
        hm = _lsb_plane_heatmap(gray)
        return DetectionResult(self.name, float(score), 0.82, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


def _flip(x: np.ndarray, kind: int) -> np.ndarray:
    x = x.astype(np.int16)
    if kind == 1:
        return (x ^ 1).astype(np.int16)
    if kind == -1:
        return (((x + 1) ^ 1) - 1).astype(np.int16)
    return x


def _disc(g: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(g.astype(np.float64)))))


def _rs_mask(channel: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    n = int(mask.size)
    flat = channel.reshape(-1)
    ngrp = (flat.size // n) * n
    g = flat[:ngrp].reshape(-1, n).astype(np.int16)
    f0 = np.sum(np.abs(np.diff(g.astype(np.float64), axis=1)), axis=1)

    def apply(kind_mask: np.ndarray) -> np.ndarray:
        out = g.copy()
        for i, m in enumerate(kind_mask):
            if m != 0:
                out[:, i] = _flip(out[:, i], int(m))
        return np.sum(np.abs(np.diff(out.astype(np.float64), axis=1)), axis=1)

    fp = apply(mask)
    fn = apply(-mask)
    rm = float(np.mean(fp > f0 + 1e-9))
    sm = float(np.mean(fp < f0 - 1e-9))
    rneg = float(np.mean(fn > f0 + 1e-9))
    sneg = float(np.mean(fn < f0 - 1e-9))
    return {"rm": rm, "sm": sm, "rneg": rneg, "sneg": sneg}


def _rs_payload_estimate(s: dict[str, float]) -> float:
    d0 = s["rm"] - s["sm"]
    d2 = s["rneg"] - s["sneg"]
    # Without a fully flipped second pass, use a linear interpolation toward zero-gap.
    denom = abs(d0) + abs(d2) + 1e-9
    p = abs(d0 - d2) / denom
    return float(np.clip(p, 0.0, 1.0))


def _lsb_plane_heatmap(gray: np.ndarray) -> np.ndarray:
    plane = (gray.astype(np.uint8) & 1).astype(np.float64)
    # Local entropy-ish: 8x8 variance of LSB.
    h, w = plane.shape
    hm = np.zeros_like(plane)
    bs = 8
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            blk = plane[y : y + bs, x : x + bs]
            hm[y : y + bs, x : x + bs] = float(np.var(blk) * 4.0)
    return np.clip(hm, 0, 1)


class SamplePairDetector(BaseDetector):
    name = "sample_pairs"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = _u8(to_gray(image))
        lsb = (gray & 1).astype(np.float64)
        if lsb.shape[1] < 3 or lsb.shape[0] < 3:
            return self.skip("image too small for sample pairs")
        hcorr = _corr(lsb[:, :-1].ravel(), lsb[:, 1:].ravel())
        vcorr = _corr(lsb[:-1, :].ravel(), lsb[1:, :].ravel())
        corr = 0.5 * (hcorr + vcorr)
        trans = 0.5 * (
            np.mean(lsb[:, :-1] != lsb[:, 1:]) + np.mean(lsb[:-1, :] != lsb[1:, :])
        )
        a = gray[:, :-1].astype(np.int16)
        b = gray[:, 1:].astype(np.int16)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        fam = (hi == lo + 1) & (lo % 2 == 0)
        if int(fam.sum()) >= 40:
            frac = float(np.mean(a[fam] % 2 == 0))
            imb = abs(frac - 0.5)
        else:
            frac, imb = 0.5, 0.0
        # Pair-family imbalance shrinks under random LSB. |corr| is extra.
        score = 0.6 * score_from_stat(imb, center=0.07, scale=0.05, invert=True) + 0.4 * score_from_stat(
            abs(corr), center=0.12, scale=0.1, invert=True
        )
        extras = {"hcorr": hcorr, "vcorr": vcorr, "lsb_transition": trans, "pair_frac_even": frac, "pair_imbalance": imb}
        expl = (
            f"Sample-pair / LSB neighbor test. corr={corr:.3f} (low => random LSB), "
            f"transition rate={trans:.3f} (0.5 => random)."
        )
        return DetectionResult(self.name, float(score), 0.8, expl, extras=extras, tier=self.tier).clamp()


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.mean(a * a) * np.mean(b * b)) + 1e-12
    return float(np.mean(a * b) / denom)


class BitplaneDetector(BaseDetector):
    name = "bitplane"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = _u8(to_gray(image))
        planes = [(b, (gray >> b) & 1) for b in range(8)]
        ents = {f"gray_b{b}": entropy(p, bins=2) for b, p in planes}
        channel_lsb = []
        for name, ch in ("r", image[..., 0]), ("g", image[..., 1]), ("b", image[..., 2]):
            e = entropy(_u8(ch) & 1, bins=2)
            ents[f"{name}_lsb"] = e
            channel_lsb.append(e)
        lsb_ent = float(np.mean(channel_lsb)) if channel_lsb else ents["gray_b0"]
        msb_ent = ents["gray_b7"]
        # Payload flattens per-channel LSB entropy toward 1 bit.
        score = score_from_stat(lsb_ent, center=0.97, scale=0.04)
        extras = ents
        expl = (
            f"Bit-plane entropy. RGB LSB H={lsb_ent:.3f} bit, gray MSB H={msb_ent:.3f}. "
            "Payloads flatten the LSB plane."
        )
        hm = (image[..., 0].astype(np.uint8) & 1).astype(np.float64)
        return DetectionResult(self.name, float(score), 0.72, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


class HistogramDetector(BaseDetector):
    name = "histogram"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = _u8(to_gray(image))
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)[:256]
        even = hist[0::2]
        odd = hist[1::2]
        pair_abs = np.abs(even - odd)
        pair_sum = even + odd + 1e-9
        rel = pair_abs / pair_sum
        # LSB mixing shrinks relative even/odd gaps on populated bins.
        populated = pair_sum > pair_sum.mean() * 0.2
        gap = float(np.mean(rel[populated])) if np.any(populated) else float(np.mean(rel))
        score = score_from_stat(gap, center=0.12, scale=0.08, invert=True)
        extras = {"even_odd_gap": gap}
        expl = f"Histogram even/odd comb gap={gap:.4f} (small => LSB-style equalization)."
        return DetectionResult(self.name, float(score), 0.7, expl, extras=extras, tier=self.tier).clamp()


class PatchworkDetector(BaseDetector):
    name = "patchwork"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        # Agnostic stand-in: signed adjacent-difference histogram should be smooth
        # for natural photos. Patchwork-like pair tweaks add extra mass at small deltas.
        dx = gray[:, 1:] - gray[:, :-1]
        hist, edges = np.histogram(dx.ravel(), bins=65, range=(-32, 32), density=True)
        center = hist[len(hist) // 2 - 2 : len(hist) // 2 + 3].sum()
        shoulders = hist[len(hist) // 2 - 8 : len(hist) // 2 - 3].sum() + hist[
            len(hist) // 2 + 3 : len(hist) // 2 + 8
        ].sum()
        ratio = float(center / (shoulders + 1e-12))
        score = score_from_stat(ratio, center=7.5, scale=2.5)
        extras = {"diff_peak_ratio": ratio}
        expl = (
            f"Patchwork-style pair test via adjacent-difference peak ratio={ratio:.3f}. "
            "Without the original key this is only a weak agnostic cue."
        )
        return DetectionResult(self.name, float(score), 0.35, expl, extras=extras, tier=self.tier).clamp()


def register_spatial() -> None:
    register(ChiSquareDetector())
    register(RSAnalysisDetector())
    register(SamplePairDetector())
    register(BitplaneDetector())
    register(HistogramDetector())
    register(PatchworkDetector())
