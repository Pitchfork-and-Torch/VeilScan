"""Eval-only synthetic watermark embedders. Not a hiding product."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from veilscan.dsp import block_view, dct2_batch, haar_dwt2, haar_idwt2, idct2_batch, tiles_to_plane

EmbedFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def _sine_cover(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Quantized sines keep structured LSB planes (good LSB-regression covers)."""
    import cv2

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = 118 + 48 * np.sin(xx / 9.5) + 18 * np.cos(yy / 13.0)
    g = 112 + 40 * np.sin(yy / 11.0) + 14 * np.sin((xx + yy) / 17.0)
    b = 104 + 44 * np.cos(xx / 8.5) + 16 * np.sin(yy / 19.0)
    base = np.stack([r, g, b], axis=-1)
    sh, sw = max(6, h // 8), max(6, w // 8)
    blob = rng.normal(0, 14, size=(sh, sw, 3)).astype(np.float32)
    blob = cv2.resize(blob, (w, h), interpolation=cv2.INTER_CUBIC)
    base = base + blob
    y0, x0 = int(h * 0.22), int(w * 0.18)
    y1, x1 = int(h * 0.58), int(w * 0.52)
    base[y0:y1, x0:x1] += rng.uniform(10, 24)
    return _u8(base)


def _photo_cover(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-ish value-noise octaves. Closer to photographic residual stats than pure sines."""
    import cv2

    acc = np.zeros((h, w, 3), dtype=np.float32)
    amp = 90.0
    for grid in (4, 8, 16, 32, 64):
        if min(h, w) < grid:
            break
        for c in range(3):
            small = rng.random((grid, grid)).astype(np.float32)
            acc[..., c] += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        amp *= 0.52
    acc -= acc.min()
    acc = acc / (acc.max() + 1e-6) * 170.0 + 28.0
    acc += rng.normal(0.0, 2.2, size=acc.shape)
    y0, x0 = int(h * 0.15), int(w * 0.55)
    y1, x1 = int(h * 0.7), int(w * 0.9)
    acc[y0:y1, x0:x1] *= rng.uniform(0.82, 0.95)
    return _u8(acc)


def synthetic_cover(
    h: int, w: int, rng: np.random.Generator, style: str = "sine"
) -> np.ndarray:
    """style: sine | photo | mix (rng chooses). Default sine keeps LSB unit tests stable."""
    if style == "mix":
        style = "sine" if rng.random() < 0.45 else "photo"
    if style == "photo":
        return _photo_cover(h, w, rng)
    return _sine_cover(h, w, rng)


def embed_lsb(
    rgb: np.ndarray,
    rng: np.random.Generator,
    rate: float = 1.0,
    sequential: bool = False,
) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    if sequential:
        n = int(np.clip(rate, 0.0, 1.0) * h * w)
        mask = np.zeros((h, w), dtype=bool)
        mask.ravel()[:n] = True
    else:
        mask = rng.random((h, w)) < rate
    payload = rng.integers(0, 2, size=(h, w), dtype=np.uint8)
    for c in range(3):
        ch = out[..., c]
        ch = (ch & 0xFE) | payload
        out[..., c] = np.where(mask, ch, out[..., c])
    return out


def embed_dct(rgb: np.ndarray, rng: np.random.Generator, amp: float = 14.0) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        blocks = block_view(out[..., c], 8, 8)
        if blocks.size == 0:
            continue
        ny, nx = blocks.shape[:2]
        coeff = dct2_batch(blocks)
        coeff[..., 3, 4] += amp * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(ny, nx))
        coeff[..., 4, 3] += amp * 0.7 * rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(ny, nx))
        rec = tiles_to_plane(idct2_batch(coeff))
        out[: rec.shape[0], : rec.shape[1], c] = rec
    return _u8(out)


def embed_dwt(rgb: np.ndarray, rng: np.random.Generator, strength: float = 0.2) -> np.ndarray:
    out = rgb.astype(np.float64).copy()
    for c in range(3):
        ll, lh, hl, hh = haar_dwt2(out[..., c])
        pattern = rng.normal(0, 1, size=hh.shape)
        scale = strength * (np.std(ll) + 1e-6)
        hh = hh + scale * pattern
        rec = haar_idwt2(ll, lh, hl, hh)
        h, w = rec.shape
        out[:h, :w, c] = rec
    return _u8(out)


def embed_svd(rgb: np.ndarray, rng: np.random.Generator, strength: float = 0.04) -> np.ndarray:
    out = rgb.astype(np.float64).copy()
    for c in range(3):
        ch = out[..., c]
        try:
            u, s, vt = np.linalg.svd(ch, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        s = s.copy()
        n = min(12, s.size)
        s[:n] = s[:n] * (1.0 + strength * rng.normal(0, 1, size=n))
        rec = (u * s) @ vt
        out[..., c] = rec
    return _u8(out)


def embed_patchwork(rgb: np.ndarray, rng: np.random.Generator, delta: float = 3.0, n_pairs: int = 4000) -> np.ndarray:
    out = rgb.astype(np.float64)
    h, w = out.shape[:2]
    n_pairs = min(n_pairs, (h * w) // 4)
    idx_a = rng.integers(0, h * w, size=n_pairs)
    idx_b = rng.integers(0, h * w, size=n_pairs)
    flat = out.reshape(-1, 3)
    flat[idx_a] = np.clip(flat[idx_a] + delta, 0, 255)
    flat[idx_b] = np.clip(flat[idx_b] - delta, 0, 255)
    return _u8(flat.reshape(out.shape))


def embed_spread(rgb: np.ndarray, rng: np.random.Generator, amp: float = 5.0) -> np.ndarray:
    """PN sequence added to DCT AC coefficients."""
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        blocks = block_view(out[..., c], 8, 8)
        if blocks.size == 0:
            continue
        ny, nx = blocks.shape[:2]
        pn = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(ny, nx))
        coeff = dct2_batch(blocks)
        coeff[..., 1:, 1:] += amp * pn[..., None, None]
        rec = tiles_to_plane(idct2_batch(coeff))
        out[: rec.shape[0], : rec.shape[1], c] = rec
    return _u8(out)


def embed_tree_ring(rgb: np.ndarray, rng: np.random.Generator, strength: float = 0.4) -> np.ndarray:
    """Concentric ring in image FFT (Tree-Ring pixel-space approximation)."""
    out = rgb.astype(np.float64).copy()
    h, w = out.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = float(rng.uniform(0.18, 0.32) * min(cy, cx))
    ring = np.exp(-0.5 * ((r - radius) / 1.6) ** 2)
    for c in range(3):
        ch = out[..., c]
        spec = np.fft.fftshift(np.fft.fft2(ch - ch.mean()))
        spec = spec * (1.0 + strength * 4.0 * ring)
        rec = np.fft.ifft2(np.fft.ifftshift(spec)).real + ch.mean()
        out[..., c] = rec
    return _u8(out)


def embed_hidden_approx(rgb: np.ndarray, rng: np.random.Generator, strength: float = 1.6) -> np.ndarray:
    """Fixed high-pass kernel residual (HiDDeN/StegaStamp-like spatial pattern)."""
    import cv2

    kernel = rng.normal(0, 1, size=(5, 5)).astype(np.float32)
    kernel -= kernel.mean()
    kernel /= np.linalg.norm(kernel) + 1e-9
    out = rgb.astype(np.float32)
    for c in range(3):
        residual = cv2.filter2D(out[..., c], -1, kernel)
        out[..., c] = np.clip(out[..., c] + strength * residual / (np.std(residual) + 1e-6), 0, 255)
    return _u8(out)


FAMILIES: dict[str, EmbedFn] = {
    "lsb": embed_lsb,
    "dct": embed_dct,
    "dwt": embed_dwt,
    "svd": embed_svd,
    "patchwork": embed_patchwork,
    "spread": embed_spread,
    "tree_ring": embed_tree_ring,
    "hidden_approx": embed_hidden_approx,
}


def embed(rgb: np.ndarray, family: str, seed: int = 0) -> np.ndarray:
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family}. known: {sorted(FAMILIES)}")
    rng = np.random.default_rng(seed)
    return FAMILIES[family](rgb, rng)
