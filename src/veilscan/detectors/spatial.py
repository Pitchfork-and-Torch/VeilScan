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
        seq_best = 0.0
        seq_frac = 1.0
        for label, ch in [("r", image[..., 0]), ("g", image[..., 1]), ("b", image[..., 2])]:
            u = _u8(ch)
            p, chi, dof = _chi_square_pairs(u)
            rgb_p.append(p)
            extras[f"{label}_p"] = p
            extras[f"{label}_chi"] = chi
            extras[f"{label}_dof"] = dof
            sp, sf = _chi_square_sequential(u)
            extras[f"{label}_seq_p"] = sp
            extras[f"{label}_seq_frac"] = sf
            if sp >= seq_best:
                seq_best, seq_frac = sp, sf
        p_gray, chi_g, dof_g = _chi_square_pairs(gray)
        extras["gray_p"] = p_gray
        extras["gray_chi"] = chi_g
        extras["gray_dof"] = dof_g
        # Global RGB mean is the random-LSB statistic. Sequential prefix only
        # boosts when the *start* of the raster is more equalized than the whole
        # (Westfeld). Short prefixes otherwise inflate p-values on covers.
        p_global = float(np.mean(rgb_p)) if rgb_p else float(p_gray)
        p = p_global
        if seq_frac <= 0.25 and seq_best > p_global + 0.20:
            p = 0.5 * p_global + 0.5 * float(seq_best)
        extras["p_global"] = p_global
        extras["p_sequential"] = seq_best
        extras["seq_best_frac"] = seq_frac
        score = float(p)
        conf = 0.85 if extras["r_dof"] >= 20 else 0.45
        expl = (
            f"Chi-square LSB pair test (Westfeld-Pfitzmann). "
            f"global p={p_global:.4f}, sequential prefix p={seq_best:.4f} at {seq_frac:.0%} "
            f"(high => equalized even/odd bins)."
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


def _chi_square_sequential(channel: np.ndarray) -> tuple[float, float]:
    """Westfeld: sequential LSB equalizes the *start* of the raster first."""
    flat = _u8(channel).ravel()
    n = int(flat.size)
    if n < 512:
        p, _, _ = _chi_square_pairs(flat)
        return p, 1.0
    best_p = 0.0
    best_frac = 1.0
    for frac in (0.10, 0.25, 0.50):
        sl = flat[: max(2048, int(n * frac))]
        if sl.size >= n:
            continue
        p, _, dof = _chi_square_pairs(sl)
        if dof >= 24 and p >= best_p:
            best_p = p
            best_frac = frac
    return float(best_p), float(best_frac)


class RSAnalysisDetector(BaseDetector):
    name = "rs_analysis"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        mask = np.array([0, 1, 1, 0], dtype=np.int8)
        hats = []
        extras: dict = {}
        channels = [("gray", _u8(to_gray(image)))]
        if image.ndim == 3 and image.shape[-1] >= 3:
            channels.extend([("r", _u8(image[..., 0])), ("g", _u8(image[..., 1])), ("b", _u8(image[..., 2]))])
        gray = channels[0][1]
        for label, ch in channels:
            p_hat, meta = rs_payload_hat(ch, mask)
            hats.append(p_hat)
            extras[f"{label}_payload_hat"] = p_hat
            extras[f"{label}_gap"] = meta.get("gap", 0.0)
        p_hat = float(np.max(hats)) if hats else 0.0
        extras["payload_hat"] = p_hat
        extras["gap"] = float(extras.get("gray_gap", 0.0))
        # Quadratic p-hat is reported but saturates on some covers; detection
        # uses the RM/SM vs R-M/S-M gap (Fridrich diagram).
        score = score_from_stat(extras["gap"], center=0.04, scale=0.06)
        expl = (
            f"RS analysis (Fridrich 2001). gap={extras['gap']:.4f}, "
            f"quadratic payload hat={p_hat:.3f}."
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


def rs_payload_hat(channel: np.ndarray, mask: np.ndarray) -> tuple[float, dict[str, float]]:
    """Fridrich-Goljan-Du quadratic payload estimate (mask M and fully LSB-flipped image)."""
    orig = _rs_mask(channel, mask)
    flipped = _u8(channel) ^ np.uint8(1)
    fl = _rs_mask(flipped, mask)
    d0 = orig["rm"] - orig["sm"]
    d1 = fl["rm"] - fl["sm"]
    d2 = orig["rneg"] - orig["sneg"]
    d3 = fl["rneg"] - fl["sneg"]
    a = 2.0 * (d1 + d0)
    b = d0 - d1 - d2 - d3
    c = d3 - d0
    linear = abs(d0 - d2) / (abs(d0) + abs(d2) + 1e-9)
    p = linear
    if abs(a) >= 1e-12:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            srt = float(np.sqrt(disc))
            minus = (-b - srt) / (2.0 * a)
            plus = (-b + srt) / (2.0 * a)
            # Fridrich: take the minus root when it lands in [0, 1].
            if 0.0 <= minus <= 1.0:
                p = float(minus)
            elif 0.0 <= plus <= 1.0:
                p = float(plus)
    meta = {
        **orig,
        "d0": float(d0),
        "d1": float(d1),
        "d2": float(d2),
        "d3": float(d3),
        "gap": float(abs(d0 - d2)),
        "payload_hat": float(np.clip(p, 0.0, 1.0)),
    }
    return meta["payload_hat"], meta


def _lsb_plane_heatmap(gray: np.ndarray) -> np.ndarray:
    plane = (gray.astype(np.uint8) & 1).astype(np.float64)
    h, w = plane.shape
    bh, bw = h - (h % 8), w - (w % 8)
    out = np.zeros_like(plane)
    if bh < 8 or bw < 8:
        return out
    blocks = plane[:bh, :bw].reshape(bh // 8, 8, bw // 8, 8).swapaxes(1, 2)
    var = np.clip(blocks.var(axis=(2, 3)) * 4.0, 0.0, 1.0)
    out[:bh, :bw] = np.repeat(np.repeat(var, 8, axis=0), 8, axis=1)
    return out


class SamplePairDetector(BaseDetector):
    name = "sample_pairs"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = _u8(to_gray(image))
        if gray.shape[1] < 3 or gray.shape[0] < 3:
            return self.skip("image too small for sample pairs")
        p_hat, imb, n_x, extras = spa_payload_hat(gray)
        extras["payload_hat"] = p_hat
        extras["pair_imbalance"] = imb
        extras["n_x"] = n_x
        score = score_from_stat(p_hat, center=0.12, scale=0.14)
        conf = 0.78 if n_x >= 80 else 0.45
        expl = (
            f"Sample pair analysis (Dumitrescu). payload hat={p_hat:.3f}, "
            f"|W-V|/X={imb:.3f} (LSB mixing equalizes directed {{2k,2k+1}} pairs)."
        )
        return DetectionResult(self.name, float(score), conf, expl, extras=extras, tier=self.tier).clamp()


def spa_payload_hat(channel: np.ndarray) -> tuple[float, float, int, dict[str, float]]:
    """Dumitrescu-style directed pair counts on {2k, 2k+1} families (horizontal+vertical)."""
    x = _u8(channel).astype(np.int32)
    u = np.concatenate([x[:, :-1].ravel(), x[:-1, :].ravel()])
    v = np.concatenate([x[:, 1:].ravel(), x[1:, :].ravel()])
    # X: unordered pair values {2k, 2k+1}
    mn = np.minimum(u, v)
    mx = np.maximum(u, v)
    in_x = (mx == mn + 1) & (mn % 2 == 0)
    n_x = int(in_x.sum())
    # Directed members of X:
    # W: (2k, 2k+1) even then odd; V: (2k+1, 2k) odd then even.
    n_w = int(((u % 2 == 0) & (v == u + 1)).sum())
    n_v = int(((u % 2 == 1) & (v == u - 1)).sum())
    imb = abs(n_w - n_v) / (n_x + 1e-9) if n_x else 0.0
    # Full random LSB drives W ~ V ~ X/2, so imb -> 0 and p_hat -> 1.
    p_hat = float(np.clip(1.0 - 2.0 * imb, 0.0, 1.0)) if n_x >= 20 else 0.0
    extras = {"n_w": float(n_w), "n_v": float(n_v), "n_x": float(n_x)}
    return p_hat, float(imb), n_x, extras


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


def patchwork_perm_null(
    gray: np.ndarray,
    *,
    n_pairs: int = 4000,
    n_perm: int = 24,
    seed: int = 20260822,
) -> dict[str, float]:
    """Keyless pairing mean vs a permutation null.

    The eval generator uses a secret random pairing. This test uses a public
    pairing from the image shape so it stays keyless. z is |S_obs - mu_null| / sd.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("gray plane required")
    flat = g.ravel()
    n = int(flat.size)
    n_pairs = int(min(max(n_pairs, 32), n // 4))
    rng = np.random.default_rng(int(seed) + n)
    idx = rng.permutation(n)
    a, b = idx[:n_pairs], idx[n_pairs : 2 * n_pairs]
    s_obs = float(flat[a].mean() - flat[b].mean())
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        shuffled = rng.permutation(flat)
        null[i] = float(shuffled[a].mean() - shuffled[b].mean())
    mu = float(null.mean())
    sd = float(null.std() + 1e-12)
    z = abs(s_obs - mu) / sd
    return {
        "s_obs": s_obs,
        "null_mean": mu,
        "null_sd": sd,
        "z": z,
        "n_pairs": float(n_pairs),
        "n_perm": float(n_perm),
    }


class PatchworkDetector(BaseDetector):
    name = "patchwork"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        extras = patchwork_perm_null(gray)
        # Keep the old adjacent-diff ratio as a diagnostic, not the score.
        dx = gray[:, 1:] - gray[:, :-1]
        hist, _edges = np.histogram(dx.ravel(), bins=65, range=(-32, 32), density=True)
        center = hist[len(hist) // 2 - 2 : len(hist) // 2 + 3].sum()
        shoulders = hist[len(hist) // 2 - 8 : len(hist) // 2 - 3].sum() + hist[
            len(hist) // 2 + 3 : len(hist) // 2 + 8
        ].sum()
        extras["diff_peak_ratio"] = float(center / (shoulders + 1e-12))
        z = float(extras["z"])
        score = score_from_stat(z, center=2.2, scale=1.1)
        expl = (
            f"Patchwork pair-mean permutation null z={z:.2f}. "
            "Keyless; secret pairings stay weak."
        )
        return DetectionResult(self.name, float(score), 0.4, expl, extras=extras, tier=self.tier).clamp()


def register_spatial() -> None:
    register(ChiSquareDetector())
    register(RSAnalysisDetector())
    register(SamplePairDetector())
    register(BitplaneDetector())
    register(HistogramDetector())
    register(PatchworkDetector())
