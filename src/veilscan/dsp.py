"""Signal-processing primitives used by detectors and generators."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def sigmoid(x: float | np.ndarray, lo: float = -30.0, hi: float = 30.0) -> float | np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), lo, hi)
    return 1.0 / (1.0 + np.exp(-z))


def to_gray(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        return rgb.astype(np.float64)
    r, g, b = rgb[..., 0].astype(np.float64), rgb[..., 1].astype(np.float64), rgb[..., 2].astype(np.float64)
    return 0.299 * r + 0.587 * g + 0.114 * b


def to_float01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def rgb_to_ycbcr(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = to_float01(rgb)
    if cv2 is not None:
        hsv = cv2.cvtColor((x * 255.0).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float64)
        return hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mx = x.max(axis=-1)
    mn = x.min(axis=-1)
    df = mx - mn + 1e-12
    v = mx
    s = np.where(mx > 0, df / (mx + 1e-12), 0.0)
    return np.zeros_like(mx), s, v


def block_view(arr: np.ndarray, bh: int, bw: int) -> np.ndarray:
    h, w = arr.shape[:2]
    h2, w2 = h - (h % bh), w - (w % bw)
    a = arr[:h2, :w2]
    return a.reshape(h2 // bh, bh, w2 // bw, bw).swapaxes(1, 2)


def dct2_block(block: np.ndarray) -> np.ndarray:
    x = np.asarray(block, dtype=np.float32)
    if cv2 is not None:
        return cv2.dct(x)
    # Orthonormal-ish separable type-II DCT via FFT (even extension).
    return _dct2_numpy(x)


def idct2_block(coeff: np.ndarray) -> np.ndarray:
    x = np.asarray(coeff, dtype=np.float32)
    if cv2 is not None:
        return cv2.idct(x)
    return _idct2_numpy(x)


def _dct_1d(a: np.ndarray, axis: int) -> np.ndarray:
    n = a.shape[axis]
    a = np.moveaxis(a, axis, -1)
    v = np.concatenate([a, a[..., ::-1]], axis=-1)
    V = np.fft.fft(v, axis=-1).real[..., :n]
    k = np.arange(n)
    V = V * (2.0 * np.cos(np.pi * k / (2.0 * n)))
    return np.moveaxis(V, -1, axis)


def _dct2_numpy(x: np.ndarray) -> np.ndarray:
    return _dct_1d(_dct_1d(x.astype(np.float64), 0), 1).astype(np.float32)


def _idct2_numpy(x: np.ndarray) -> np.ndarray:
    # Approximate inverse matching the unnormalized type-II used above.
    n0, n1 = x.shape
    y = x.astype(np.float64).copy()
    k0 = np.arange(n0)[:, None]
    k1 = np.arange(n1)[None, :]
    y = y / np.clip(2.0 * np.cos(np.pi * k0 / (2.0 * n0)), 1e-6, None)
    y = y / np.clip(2.0 * np.cos(np.pi * k1 / (2.0 * n1)), 1e-6, None)
    v = np.concatenate([y, y[::-1, :]], axis=0)
    s = np.fft.ifft(v, axis=0).real[:n0]
    v2 = np.concatenate([s, s[:, ::-1]], axis=1)
    return np.fft.ifft(v2, axis=1).real[:, :n1].astype(np.float32)


def haar_dwt2(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One-level 2D Haar. Returns LL, LH, HL, HH."""
    x = np.asarray(img, dtype=np.float64)
    h, w = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    x = x[:h, :w]
    even_c = x[:, 0::2]
    odd_c = x[:, 1::2]
    s = (even_c + odd_c) / np.sqrt(2.0)
    d = (even_c - odd_c) / np.sqrt(2.0)
    ll = (s[0::2] + s[1::2]) / np.sqrt(2.0)
    hl = (s[0::2] - s[1::2]) / np.sqrt(2.0)
    lh = (d[0::2] + d[1::2]) / np.sqrt(2.0)
    hh = (d[0::2] - d[1::2]) / np.sqrt(2.0)
    return ll, lh, hl, hh


def haar_idwt2(ll: np.ndarray, lh: np.ndarray, hl: np.ndarray, hh: np.ndarray) -> np.ndarray:
    s_even = (ll + hl) / np.sqrt(2.0)
    s_odd = (ll - hl) / np.sqrt(2.0)
    d_even = (lh + hh) / np.sqrt(2.0)
    d_odd = (lh - hh) / np.sqrt(2.0)
    h, w = ll.shape
    s = np.empty((h * 2, w), dtype=np.float64)
    d = np.empty((h * 2, w), dtype=np.float64)
    s[0::2], s[1::2] = s_even, s_odd
    d[0::2], d[1::2] = d_even, d_odd
    out = np.empty((h * 2, w * 2), dtype=np.float64)
    out[:, 0::2] = (s + d) / np.sqrt(2.0)
    out[:, 1::2] = (s - d) / np.sqrt(2.0)
    return out


def radial_profile(mag: np.ndarray) -> np.ndarray:
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    rmax = int(r.max()) + 1
    sums = np.bincount(r.ravel(), mag.ravel(), minlength=rmax)
    counts = np.bincount(r.ravel(), minlength=rmax).astype(np.float64)
    return sums / np.maximum(counts, 1.0)


def fft_shift_mag(gray: np.ndarray) -> np.ndarray:
    x = np.asarray(gray, dtype=np.float64)
    x = x - x.mean()
    spec = np.fft.fftshift(np.fft.fft2(x))
    return np.log1p(np.abs(spec))


def highpass_residual(gray: np.ndarray, ksize: int = 5) -> np.ndarray:
    x = np.asarray(gray, dtype=np.float64)
    if cv2 is not None:
        blur = cv2.GaussianBlur(x, (ksize, ksize), 0)
    else:
        blur = _box_blur(x, ksize)
    return x - blur


def _box_blur(x: np.ndarray, k: int) -> np.ndarray:
    k = max(3, k | 1)
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    c = np.cumsum(np.cumsum(xp, 0), 1)
    h, w = x.shape
    out = (
        c[k : k + h, k : k + w]
        - c[:h, k : k + w]
        - c[k : k + h, :w]
        + c[:h, :w]
    )
    return out / float(k * k)


def kv_residual(gray: np.ndarray) -> np.ndarray:
    """KV high-pass used in SRM-like residual analysis."""
    kernel = np.array(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=np.float64,
    )
    x = np.asarray(gray, dtype=np.float64)
    if cv2 is not None:
        return cv2.filter2D(x, -1, kernel, borderType=cv2.BORDER_REFLECT)
    # naive conv fallback
    pad = 2
    xp = np.pad(x, pad, mode="reflect")
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.sum(xp[i : i + 5, j : j + 5] * kernel)
    return out


def entropy(x: np.ndarray, bins: int = 256) -> float:
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist[hist > 0]
    return float(-(p * np.log2(p)).sum())


def kurtosis(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=np.float64).ravel()
    z = z - z.mean()
    m2 = np.mean(z ** 2) + 1e-12
    m4 = np.mean(z ** 4)
    return float(m4 / (m2 ** 2) - 3.0)


def skewness(x: np.ndarray) -> float:
    z = np.asarray(x, dtype=np.float64).ravel()
    z = z - z.mean()
    m2 = np.mean(z ** 2) + 1e-12
    m3 = np.mean(z ** 3)
    return float(m3 / (m2 ** 1.5))


def laplacian_negloglike(coeff: np.ndarray) -> float:
    """Higher => less Laplacian (more structured / extra energy)."""
    z = np.asarray(coeff, dtype=np.float64).ravel()
    z = z[np.isfinite(z)]
    if z.size < 16:
        return 0.0
    b = np.mean(np.abs(z - np.median(z))) + 1e-12
    nll = np.log(2.0 * b) + np.mean(np.abs(z - np.median(z))) / b
    gauss_nll = 0.5 * np.log(2.0 * np.pi * (np.var(z) + 1e-12)) + 0.5
    return float(nll - gauss_nll)


def score_from_stat(stat: float, center: float, scale: float, invert: bool = False) -> float:
    z = (stat - center) / max(scale, 1e-9)
    if invert:
        z = -z
    return float(sigmoid(z))
