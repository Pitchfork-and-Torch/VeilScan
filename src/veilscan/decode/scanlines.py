"""Scan each raster row for RGB/R/G/B LSB headers. Finds short patch payloads."""

from __future__ import annotations

import numpy as np

from veilscan.decode.lsb import pack_bits
from veilscan.decode.score import extract_run_candidates, score_text, utf8_text


def iter_scanline_hits(rgb: np.ndarray) -> list[tuple[str, str, float, tuple[int, int, int, int]]]:
    """Return (layout, text, score, bbox) for token/text hits along rows."""
    img = np.asarray(rgb)
    if img.ndim != 3 or img.shape[2] < 3:
        return []
    h, w = img.shape[:2]
    hits: list[tuple[str, str, float, tuple[int, int, int, int]]] = []
    interleaved = _interleave(img)
    planes = [
        ("rgb-bit0-msb", interleaved, True),
        ("rgb-bit0-lsb", interleaved, False),
        ("r-bit0-msb", img[..., 0] & 1, True),
        ("g-bit0-msb", img[..., 1] & 1, True),
        ("b-bit0-msb", img[..., 2] & 1, True),
    ]
    for layout, bits2d, msb in planes:
        for y in range(h):
            row = bits2d[y].astype(np.uint8).ravel()
            for off in range(8):
                raw = pack_bits(row[off:], msb_first=msb)
                for payload, framed in extract_run_candidates(raw):
                    text = utf8_text(payload)
                    if text is None:
                        continue
                    s = score_text(text, framed=framed)
                    if s <= 0:
                        continue
                    hits.append((layout, text, s, (0, y, w, 1)))
                    if s >= 0.93 and framed and len(text) >= 8:
                        return hits
    return hits


def _interleave(img: np.ndarray) -> np.ndarray:
    r = img[..., 0] & 1
    g = img[..., 1] & 1
    b = img[..., 2] & 1
    h, w = r.shape
    out = np.empty((h, w * 3), dtype=np.uint8)
    out[:, 0::3] = r
    out[:, 1::3] = g
    out[:, 2::3] = b
    return out
