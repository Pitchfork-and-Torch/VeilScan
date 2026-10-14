"""QR / barcode recovery from an LSB bit plane. Uses OpenCV."""

from __future__ import annotations

import numpy as np


def _lsb_plane(rgb: np.ndarray, channel: int | None) -> np.ndarray:
    if channel is None:
        plane = (rgb[..., 0] & 1) | (rgb[..., 1] & 1) | (rgb[..., 2] & 1)
    else:
        plane = rgb[..., channel] & 1
    return (plane * 255).astype(np.uint8)


def decode_lsb_qr(rgb: np.ndarray) -> list[str]:
    try:
        import cv2
    except Exception:
        return []
    if rgb is None or rgb.size == 0 or min(rgb.shape[:2]) < 16:
        return []
    det = cv2.QRCodeDetector()
    found: list[str] = []
    seen: set[str] = set()
    h, w = rgb.shape[:2]
    scales = (1, 2, 4) if max(h, w) <= 400 else (1, 2)
    for ch in (0, 1, 2, None):
        plane = _lsb_plane(rgb, ch)
        for inv in (False, True):
            img = 255 - plane if inv else plane
            for sc in scales:
                if sc == 1:
                    im = img
                else:
                    im = cv2.resize(img, (w * sc, h * sc), interpolation=cv2.INTER_NEAREST)
                try:
                    val, _pts, _ = det.detectAndDecode(im)
                except Exception:
                    continue
                if val and val not in seen:
                    seen.add(val)
                    found.append(val)
    return found
