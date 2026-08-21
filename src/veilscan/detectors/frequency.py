"""Transform-domain detectors: DCT, DFT, DWT, hybrid, Tree-Ring spectral, Fourier-Mellin."""

from __future__ import annotations

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.dsp import (
    block_view,
    dct2_block,
    fft_shift_mag,
    haar_dwt2,
    kurtosis,
    laplacian_negloglike,
    radial_profile,
    score_from_stat,
    to_gray,
)
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult


class DCTDetector(BaseDetector):
    name = "dct"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        blocks = block_view(gray, 8, 8)
        if blocks.size == 0:
            return self.skip("too small for 8x8 DCT")
        nb_y, nb_x, _, _ = blocks.shape
        mid_energy = np.zeros((nb_y, nb_x), dtype=np.float64)
        mid_coeffs = []
        for i in range(nb_y):
            for j in range(nb_x):
                c = dct2_block(blocks[i, j])
                mask = _mid_band_mask(8)
                mid = c[mask]
                mid_energy[i, j] = float(np.mean(mid ** 2))
                mid_coeffs.append(mid.ravel())
        mid_all = np.concatenate(mid_coeffs)
        e = float(np.mean(mid_energy))
        k = kurtosis(mid_all)
        lap = laplacian_negloglike(mid_all)
        # Extra mid-band energy and less-Laplacian AC => watermark-like.
        score = (
            0.55 * score_from_stat(np.log1p(e), center=0.95, scale=0.35)
            + 0.30 * score_from_stat(k, center=40.0, scale=18.0, invert=True)
            + 0.15 * score_from_stat(lap, center=-0.4, scale=0.25)
        )
        hm = np.repeat(np.repeat(mid_energy / (mid_energy.max() + 1e-9), 8, 0), 8, 1)
        extras = {"mid_energy": e, "kurtosis": k, "laplacian_nll_gap": lap}
        expl = (
            f"Block DCT mid-band. energy={e:.3f}, kurtosis={k:.2f}, "
            f"Laplacian gap={lap:.3f}. Robust classical marks live here."
        )
        return DetectionResult(self.name, float(score), 0.8, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


def _mid_band_mask(n: int) -> np.ndarray:
    u = np.arange(n)[:, None]
    v = np.arange(n)[None, :]
    s = u + v
    return (s >= 4) & (s <= 10) & ~((u == 0) & (v == 0))


class DFTDetector(BaseDetector):
    name = "dft"
    tier = "fast"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        mag = fft_shift_mag(gray)
        rad = radial_profile(np.expm1(mag))
        if rad.size < 16:
            return self.skip("spectrum too small")
        # Fit log-log 1/f in the mid radial band; residual at high freq is the cue.
        r = np.arange(1, rad.size)
        y = np.log(rad[1:] + 1e-12)
        x = np.log(r.astype(np.float64))
        lo, hi = max(4, rad.size // 16), max(8, rad.size // 2)
        sl = slice(lo, hi)
        A = np.vstack([x[sl], np.ones(hi - lo)]).T
        slope, intercept = np.linalg.lstsq(A, y[sl], rcond=None)[0]
        pred = slope * x + intercept
        resid = y - pred
        high = resid[int(0.65 * resid.size) :]
        excess = float(np.mean(high))
        score = score_from_stat(excess, center=0.15, scale=0.35)
        extras = {"one_over_f_slope": float(slope), "high_freq_excess": excess}
        expl = (
            f"DFT radial 1/f fit. slope={slope:.3f} (photos ~ -1 to -3), "
            f"high-freq residual={excess:.3f}."
        )
        hm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-9)
        return DetectionResult(self.name, float(score), 0.65, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


class DWTDetector(BaseDetector):
    name = "dwt"
    tier = "frequency"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        ll, lh, hl, hh = haar_dwt2(gray)
        e_ll = float(np.mean(ll ** 2) + 1e-12)
        e_hh = float(np.mean(hh ** 2))
        e_lh = float(np.mean(lh ** 2))
        e_hl = float(np.mean(hl ** 2))
        ratio = (e_hh + e_lh + e_hl) / e_ll
        k_hh = kurtosis(hh)
        score = 0.7 * score_from_stat(np.log1p(ratio * 1000.0), center=0.8, scale=0.7) + 0.3 * score_from_stat(
            k_hh, center=1.5, scale=3.0
        )
        extras = {"hh_ll": e_hh / e_ll, "detail_ll": ratio, "hh_kurtosis": k_hh}
        hm = np.clip(np.abs(hh) / (np.percentile(np.abs(hh), 99) + 1e-9), 0, 1)
        expl = (
            f"Haar DWT. detail/LL energy={ratio:.4f}, HH kurtosis={k_hh:.2f}. "
            "DWT marks inflate high-frequency subbands."
        )
        return DetectionResult(self.name, float(score), 0.78, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


class HybridDDSDetector(BaseDetector):
    name = "hybrid_dds"
    tier = "frequency"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        ll, lh, hl, hh = haar_dwt2(gray)
        bands = []
        for band in (lh, hl):
            blocks = block_view(band, 8, 8)
            if blocks.size == 0:
                continue
            for i in range(blocks.shape[0]):
                for j in range(blocks.shape[1]):
                    c = dct2_block(blocks[i, j])
                    try:
                        s = np.linalg.svd(c, compute_uv=False)
                    except np.linalg.LinAlgError:
                        continue
                    if s.size >= 2 and s[0] > 1e-9:
                        bands.append(float(s[1] / s[0]))
        if not bands:
            return self.skip("not enough DWT-DCT blocks")
        ratio = float(np.mean(bands))
        score = score_from_stat(ratio, center=0.48, scale=0.08)
        extras = {"mean_sigma2_over_sigma1": ratio, "n_blocks": len(bands)}
        expl = (
            f"Hybrid DWT-DCT-SVD (DDS-style). mean σ2/σ1={ratio:.4f}. "
            "SVD tweaks flatten the singular spectrum of mid-band blocks."
        )
        return DetectionResult(self.name, float(score), 0.7, expl, extras=extras, tier=self.tier).clamp()


class TreeRingSpectralDetector(BaseDetector):
    name = "tree_ring_spectral"
    tier = "frequency"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        x = gray - gray.mean()
        spec = np.fft.fftshift(np.fft.fft2(x))
        mag = np.abs(spec)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = int(min(cy, cx) * 0.85)
        r_lo = max(10, int(0.16 * rmax))
        if rmax - r_lo < 8:
            return self.skip("image too small for ring analysis")
        rad = []
        circ_cv = []
        radii = []
        for rr in range(r_lo, rmax):
            band = (r >= rr - 0.6) & (r < rr + 0.6)
            vals = mag[band]
            if vals.size < 16:
                continue
            radii.append(rr)
            rad.append(float(vals.mean()))
            circ_cv.append(float(vals.std() / (vals.mean() + 1e-9)))
        if len(rad) < 8:
            return self.skip("not enough radii")
        rad_a = np.asarray(rad)
        cv_a = np.asarray(circ_cv)
        log_r = np.log(np.asarray(radii, dtype=np.float64) + 1e-6)
        log_e = np.log(rad_a + 1e-12)
        A = np.vstack([log_r, np.ones(len(radii))]).T
        slope, intercept = np.linalg.lstsq(A, log_e, rcond=None)[0]
        pred = np.exp(slope * log_r + intercept)
        excess = (rad_a - pred) / (pred + 1e-12)
        # Rings: energy excess AND low angular CV at the same radius.
        cue = excess / (cv_a + 0.08)
        idx = int(np.argmax(cue))
        ring_score = float(cue[idx])
        best_r = int(radii[idx])
        min_cv = float(cv_a.min())
        peak_excess = float(excess[idx])
        score = 0.45 * score_from_stat(min_cv, center=0.42, scale=0.14, invert=True) + 0.55 * score_from_stat(
            peak_excess, center=1.15, scale=0.45
        )
        extras = {
            "best_radius": best_r,
            "ring_cue": ring_score,
            "min_angular_cv": min_cv,
            "peak_excess": peak_excess,
        }
        expl = (
            f"Tree-Ring-style circular FFT test. strongest ring radius={best_r}px, "
            f"cue={ring_score:.4f}. Pixel-space proxy; not DDIM inversion."
        )
        hm = np.zeros_like(mag, dtype=np.float64)
        band = (r >= best_r - 1.5) & (r <= best_r + 1.5)
        hm[band] = 1.0
        hm = hm / (hm.max() + 1e-9)
        return DetectionResult(self.name, float(score), 0.7, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


class FourierMellinDetector(BaseDetector):
    name = "fourier_mellin"
    tier = "frequency"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        mag = fft_shift_mag(gray)
        h, w = mag.shape
        cy, cx = h / 2.0, w / 2.0
        # Log-polar resample of magnitude (cheap geometric invariant).
        n_r, n_a = 64, 128
        r = np.logspace(0, np.log10(min(cy, cx) * 0.9 + 1e-6), n_r)
        ang = np.linspace(0, 2 * np.pi, n_a, endpoint=False)
        rr, aa = np.meshgrid(r, ang, indexing="ij")
        yy = cy + rr * np.sin(aa)
        xx = cx + rr * np.cos(aa)
        # bilinear
        x0 = np.clip(np.floor(xx).astype(int), 0, w - 2)
        y0 = np.clip(np.floor(yy).astype(int), 0, h - 2)
        dx = xx - x0
        dy = yy - y0
        lp = (
            mag[y0, x0] * (1 - dx) * (1 - dy)
            + mag[y0, x0 + 1] * dx * (1 - dy)
            + mag[y0 + 1, x0] * (1 - dx) * dy
            + mag[y0 + 1, x0 + 1] * dx * dy
        )
        # Energy concentration along log-radius (scale-invariant rings).
        prof = lp.mean(axis=1)
        prof = prof - np.convolve(prof, np.ones(5) / 5.0, mode="same")
        peak = float(np.max(np.abs(prof)))
        score = score_from_stat(peak, center=4.2, scale=1.4)
        extras = {"logpolar_peak": peak}
        expl = f"Fourier-Mellin log-polar peak={peak:.4f} (rotation/scale tolerant spectral bumps)."
        return DetectionResult(self.name, float(score), 0.35, expl, extras=extras, tier=self.tier).clamp()


class BlockMultiscaleDetector(BaseDetector):
    name = "block_multiscale"
    tier = "frequency"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        gray = to_gray(image)
        from veilscan.dsp import highpass_residual

        res = highpass_residual(gray, 5)
        maps = []
        for bs in (8, 16, 32):
            if min(gray.shape) < bs * 2:
                continue
            blocks = block_view(res, bs, bs)
            e = np.mean(blocks ** 2, axis=(2, 3))
            maps.append(e)
        if not maps:
            return self.skip("too small for multi-scale blocks")
        # Structured block energy (low spatial entropy of energy map) can be a tiled mark.
        e0 = maps[0]
        spatial_var = float(np.var(e0) / (np.mean(e0) ** 2 + 1e-12))
        score = score_from_stat(spatial_var, center=3.0, scale=1.8)
        extras = {"energy_cv2": spatial_var, "n_scales": len(maps)}
        hm = e0 / (e0.max() + 1e-9)
        hm = np.repeat(np.repeat(hm, 8, 0), 8, 1)
        expl = f"Multi-scale block residual energy CV^2={spatial_var:.3f}."
        return DetectionResult(self.name, float(score), 0.55, expl, heatmap=hm, extras=extras, tier=self.tier).clamp()


def register_frequency() -> None:
    register(DCTDetector())
    register(DFTDetector())
    register(DWTDetector())
    register(HybridDDSDetector())
    register(TreeRingSpectralDetector())
    register(FourierMellinDetector())
    register(BlockMultiscaleDetector())
