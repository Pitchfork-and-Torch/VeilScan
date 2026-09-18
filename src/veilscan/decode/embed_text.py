"""Eval-only plaintext planter. Not a hiding product."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from veilscan.decode.container import (
    plant_jpeg_com,
    plant_png_itxt,
    plant_png_text,
    plant_png_ztxt,
)
from veilscan.decode.layouts import DEFAULT_LAYOUT_ID, get_layout
from veilscan.decode.lsb import message_to_bits, plant_bits


def embed_text_patch(
    rgb: np.ndarray,
    message: str,
    x: int,
    y: int,
    w: int,
    h: int,
    layout_id: str = DEFAULT_LAYOUT_ID,
) -> np.ndarray:
    """Eval-only: plant sequential LSB in a rectangle. Tests localized decode."""
    out = np.ascontiguousarray(rgb.copy())
    crop = out[y : y + h, x : x + w]
    marked = embed_text_array(crop, message, layout_id=layout_id)
    out[y : y + h, x : x + w] = marked
    return out


def embed_text_array(
    rgb: np.ndarray,
    message: str,
    layout_id: str = DEFAULT_LAYOUT_ID,
) -> np.ndarray:
    layout = get_layout(layout_id)
    bits = message_to_bits(message, msb_first=bool(layout["msb_first"]))
    return plant_bits(rgb, bits, layout)


def embed_text_png_bytes(
    rgb: np.ndarray,
    message: str,
    layout_id: str = DEFAULT_LAYOUT_ID,
) -> bytes:
    marked = embed_text_array(rgb, message, layout_id=layout_id)
    buf = BytesIO()
    Image.fromarray(marked).save(buf, format="PNG", optimize=False, compress_level=6)
    return buf.getvalue()


def embed_text_path(
    inp: str | Path,
    out: str | Path,
    message: str,
    layout_id: str = DEFAULT_LAYOUT_ID,
    family: str = "lsb",
) -> Path:
    """family: lsb | png-text | png-ztxt | png-itxt | jpeg-com. lsb writes PNG."""
    inp = Path(inp)
    out = Path(out)
    im = Image.open(inp)
    if im.mode != "RGB":
        im = im.convert("RGB")
    rgb = np.asarray(im, dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    if family == "lsb":
        data = embed_text_png_bytes(rgb, message, layout_id=layout_id)
        out.write_bytes(data)
        return out
    if family == "png-text":
        out.write_bytes(plant_png_text(rgb, message))
        return out
    if family == "png-ztxt":
        out.write_bytes(plant_png_ztxt(rgb, message))
        return out
    if family == "png-itxt":
        out.write_bytes(plant_png_itxt(rgb, message))
        return out
    if family == "jpeg-com":
        out.write_bytes(plant_jpeg_com(rgb, message))
        return out
    raise KeyError(
        f"unknown embed-text family {family}. known: lsb, png-text, png-ztxt, png-itxt, jpeg-com"
    )
