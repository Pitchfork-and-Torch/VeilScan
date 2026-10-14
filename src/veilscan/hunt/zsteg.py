"""zsteg-class LSB walks. Native numpy. No Ruby."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from veilscan.decode.lsb import pack_bits
from veilscan.hunt.payloads import inspect_payload
from veilscan.hunt.types import HuntFinding

_CHANS = ("r", "g", "b", "a", "rgb", "bgr", "rgba")
_ORDERS = ("row", "col", "snake")


def _order_bits(plane: np.ndarray, order: str) -> np.ndarray:
    """plane is HxW uint8 bits {0,1}."""
    if order == "col":
        return np.ascontiguousarray(plane.T).ravel()
    if order == "snake":
        rows = []
        for i, row in enumerate(plane):
            rows.append(row[::-1] if i % 2 else row)
        return np.concatenate(rows) if rows else plane.ravel()
    return plane.ravel()


def _bit_plane(ch: np.ndarray, bit: int) -> np.ndarray:
    return ((ch >> bit) & 1).astype(np.uint8)


def _walk_bits(rgba: np.ndarray, walk: str, bit: int, order: str) -> np.ndarray | None:
    h, w, c = rgba.shape
    r = _bit_plane(rgba[:, :, 0], bit)
    g = _bit_plane(rgba[:, :, 1], bit)
    b = _bit_plane(rgba[:, :, 2], bit)
    a = _bit_plane(rgba[:, :, 3], bit) if c >= 4 else None
    if walk == "r":
        return _order_bits(r, order)
    if walk == "g":
        return _order_bits(g, order)
    if walk == "b":
        return _order_bits(b, order)
    if walk == "a":
        if a is None:
            return None
        return _order_bits(a, order)
    if walk == "rgb":
        stacked = np.stack([r, g, b], axis=-1)
        if order == "col":
            stacked = np.ascontiguousarray(stacked.transpose(1, 0, 2))
        elif order == "snake":
            rows = []
            for i in range(h):
                row = stacked[i]
                rows.append(row[::-1] if i % 2 else row)
            return np.concatenate([x.ravel() for x in rows])
        return stacked.ravel()
    if walk == "bgr":
        stacked = np.stack([b, g, r], axis=-1)
        if order == "col":
            stacked = np.ascontiguousarray(stacked.transpose(1, 0, 2))
        elif order == "snake":
            rows = []
            for i in range(h):
                row = stacked[i]
                rows.append(row[::-1] if i % 2 else row)
            return np.concatenate([x.ravel() for x in rows])
        return stacked.ravel()
    if walk == "rgba":
        if a is None:
            return None
        stacked = np.stack([r, g, b, a], axis=-1)
        if order == "col":
            stacked = np.ascontiguousarray(stacked.transpose(1, 0, 2))
        elif order == "snake":
            rows = []
            for i in range(h):
                row = stacked[i]
                rows.append(row[::-1] if i % 2 else row)
            return np.concatenate([x.ravel() for x in rows])
        return stacked.ravel()
    return None


def iter_zsteg_findings(
    rgba: np.ndarray,
    cre,
    *,
    deep: bool = False,
    stop_on_flag: bool = False,
) -> Iterator[HuntFinding]:
    if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 3:
        return
    bits = range(0, 8) if deep else range(0, 4)
    has_a = rgba.shape[2] >= 4
    seen_text: set[str] = set()
    n_out = 0
    for walk in _CHANS:
        if walk in {"a", "rgba"} and not has_a:
            continue
        for bit in bits:
            for order in _ORDERS:
                for msb_first in (True, False):
                    packed_bits = _walk_bits(rgba, walk, bit, order)
                    if packed_bits is None or packed_bits.size < 16:
                        continue
                    raw = pack_bits(packed_bits, msb_first=msb_first)
                    if not raw:
                        continue
                    layout = f"{walk}-b{bit}-{'msb' if msb_first else 'lsb'}-{order}"
                    for hit in inspect_payload(raw, cre):
                        text = str(hit.get("text") or "")
                        flags = list(hit.get("flags") or [])
                        key = text[:200] + layout
                        if key in seen_text:
                            continue
                        seen_text.add(key)
                        n_out += 1
                        yield HuntFinding(
                            family="zsteg",
                            method=layout,
                            confidence=float(hit.get("confidence") or 0.8),
                            evidence=str(hit.get("kind") or "bitstream"),
                            text=text or None,
                            flag_hit=bool(flags),
                            extra={"flags": flags, "kind": hit.get("kind"), "name": hit.get("name")},
                        )
                        if stop_on_flag and flags:
                            return
                        if n_out >= 48:
                            return
