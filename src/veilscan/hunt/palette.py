"""Palette-index LSB before PIL expands P-mode to RGB."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from PIL import Image

from veilscan.decode.lsb import pack_bits
from veilscan.hunt.payloads import inspect_payload
from veilscan.hunt.types import HuntFinding
from veilscan.hunt.zsteg import _order_bits


def index_array(im: Image.Image) -> np.ndarray | None:
    if im.mode != "P":
        return None
    return np.array(im, dtype=np.uint8)


def iter_palette_findings(
    index: np.ndarray,
    cre,
    *,
    deep: bool = False,
    stop_on_flag: bool = False,
) -> Iterator[HuntFinding]:
    if index is None or index.ndim != 2 or index.size < 16:
        return
    bits = range(0, 8) if deep else range(0, 4)
    seen: set[str] = set()
    n_out = 0
    for bit in bits:
        plane = ((index >> bit) & 1).astype(np.uint8)
        for order in ("row", "col", "snake"):
            for msb_first in (True, False):
                raw = pack_bits(_order_bits(plane, order), msb_first=msb_first)
                if not raw:
                    continue
                layout = f"palette-b{bit}-{'msb' if msb_first else 'lsb'}-{order}"
                for hit in inspect_payload(raw, cre):
                    text = str(hit.get("text") or "")
                    flags = list(hit.get("flags") or [])
                    key = text[:200] + layout
                    if key in seen:
                        continue
                    seen.add(key)
                    n_out += 1
                    yield HuntFinding(
                        family="palette",
                        method=layout,
                        confidence=float(hit.get("confidence") or 0.8),
                        evidence=str(hit.get("kind") or "index-lsb"),
                        text=text or None,
                        flag_hit=bool(flags),
                        extra={"flags": flags, "kind": hit.get("kind")},
                    )
                    if stop_on_flag and flags:
                        return
                    if n_out >= 24:
                        return
