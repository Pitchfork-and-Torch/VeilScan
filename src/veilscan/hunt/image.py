"""Load pixels for hunt without destroying palette indices or alpha."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps


def load_hunt_image(data: bytes) -> tuple[np.ndarray | None, np.ndarray | None, list[str]]:
    notes: list[str] = []
    try:
        im = Image.open(BytesIO(data))
        im = ImageOps.exif_transpose(im) or im
    except Exception as exc:  # noqa: BLE001
        return None, None, [f"pixel load failed: {exc}"]

    index = None
    if im.mode == "P":
        index = np.array(im, dtype=np.uint8)

    if im.mode == "RGBA":
        rgba = np.array(im, dtype=np.uint8)
    elif im.mode == "RGB":
        rgb = np.array(im, dtype=np.uint8)
        ones = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        rgba = np.dstack([rgb, ones])
    elif im.mode == "P":
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
        notes.append("Paletted: index LSB is separate from this RGBA expand.")
    elif im.mode == "L":
        g = np.array(im, dtype=np.uint8)
        ones = np.full_like(g, 255)
        rgba = np.stack([g, g, g, ones], axis=-1)
    elif im.mode == "LA":
        la = np.array(im, dtype=np.uint8)
        g, a = la[:, :, 0], la[:, :, 1]
        rgba = np.dstack([g, g, g, a])
    else:
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
        notes.append(f"converted {im.mode} to RGBA")

    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    return index, rgba, notes
