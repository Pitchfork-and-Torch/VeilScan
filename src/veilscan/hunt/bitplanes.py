"""Stegsolve-class bitplane sheet plus LSB-plane QR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from veilscan.decode.bitplane_qr import decode_lsb_qr
from veilscan.hunt.flags import find_flags
from veilscan.hunt.types import HuntFinding

_CH_NAMES = ("R", "G", "B", "A")


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def render_bitplane_sheet(
    arr: np.ndarray,
    dest: Path,
    *,
    max_cell: int = 128,
) -> Path:
    """arr is HxW (palette) or HxWxC. Bit 7 left, bit 0 right. One row per channel."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        planes = arr[:, :, None]
        names = ("P",)
    else:
        planes = arr
        names = _CH_NAMES[: planes.shape[2]]
    h, w = planes.shape[:2]
    scale = min(1.0, max_cell / max(h, w))
    cw = max(16, int(w * scale))
    ch = max(16, int(h * scale))
    pad = 4
    label_h = 14
    cols, rows = 8, len(names)
    sheet_w = cols * (cw + pad) + pad
    sheet_h = rows * (ch + pad + label_h) + pad
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 18, 22))
    draw = ImageDraw.Draw(sheet)
    font = _font()
    for ri, name in enumerate(names):
        chan = planes[:, :, ri]
        for bit in range(7, -1, -1):
            ci = 7 - bit
            plane = ((chan >> bit) & 1).astype(np.uint8) * 255
            im = Image.fromarray(plane, mode="L").resize((cw, ch), Image.Resampling.NEAREST)
            x = pad + ci * (cw + pad)
            y = pad + ri * (ch + pad + label_h) + label_h
            sheet.paste(im.convert("RGB"), (x, y))
            draw.text((x, y - label_h), f"{name}{bit}", fill=(180, 190, 200), font=font)
    sheet.save(dest, "PNG", optimize=True)
    return dest


def qr_findings(rgba: np.ndarray, cre) -> list[HuntFinding]:
    out: list[HuntFinding] = []
    if rgba is None or rgba.ndim != 3 or min(rgba.shape[:2]) < 16:
        return out
    seen: set[str] = set()
    c = min(rgba.shape[2], 4)
    for ch in range(c):
        for bit in (0, 1):
            plane = ((rgba[:, :, ch] >> bit) & 1).astype(np.uint8)
            fake = np.zeros((*plane.shape, 3), dtype=np.uint8)
            fake[:, :, 0] = plane * 255
            for qtext in decode_lsb_qr(fake):
                if qtext in seen:
                    continue
                seen.add(qtext)
                hits = find_flags(qtext, cre)
                out.append(
                    HuntFinding(
                        family="qr",
                        method=f"bitplane-{_CH_NAMES[ch]}{bit}",
                        confidence=0.97 if hits else 0.9,
                        evidence="OpenCV QR on bit plane",
                        text=qtext,
                        flag_hit=bool(hits),
                        extra={"flags": hits},
                    )
                )
    if rgba.shape[2] >= 3:
        for qtext in decode_lsb_qr(rgba[:, :, :3]):
            if qtext in seen:
                continue
            seen.add(qtext)
            hits = find_flags(qtext, cre)
            out.append(
                HuntFinding(
                    family="qr",
                    method="lsb-rgb",
                    confidence=0.97 if hits else 0.9,
                    evidence="OpenCV QR on RGB LSB",
                    text=qtext,
                    flag_hit=bool(hits),
                    extra={"flags": hits},
                )
            )
    return out
