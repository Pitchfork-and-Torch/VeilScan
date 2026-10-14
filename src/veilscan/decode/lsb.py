"""Sequential LSB extract / plant on raw HxWxC uint8 (no canvas, no JPEG decode)."""

from __future__ import annotations

import numpy as np

from veilscan.decode.layouts import LsbLayout

_WALK_INDEX = {"r": 0, "g": 1, "b": 2}


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("LSB needs at least 3 channels (RGB or RGBA)")
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return arr


def extract_bits(image: np.ndarray, layout: LsbLayout) -> np.ndarray:
    rgb = _as_rgb(image)
    pix = rgb.reshape(-1, rgb.shape[2])
    bit = int(layout["bit"])
    walk = layout["walk"]
    mask = np.uint8(1 << bit)

    def ch(idx: int) -> np.ndarray:
        return ((pix[:, idx] & mask) >> bit).astype(np.uint8)

    if walk in _WALK_INDEX:
        return ch(_WALK_INDEX[walk])
    r, g, b = ch(0), ch(1), ch(2)
    if walk == "rgb":
        out = np.empty(r.size * 3, dtype=np.uint8)
        out[0::3] = r
        out[1::3] = g
        out[2::3] = b
        return out
    if walk == "rgb-planar":
        return np.concatenate([r, g, b])
    raise ValueError(f"unknown walk {walk}")


def pack_bits(bits: np.ndarray, msb_first: bool) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    n = (bits.size // 8) * 8
    if n == 0:
        return b""
    grouped = bits[:n].reshape(-1, 8)
    if msb_first:
        weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint16)
    else:
        weights = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint16)
    return (grouped.astype(np.uint16) * weights).sum(axis=1).astype(np.uint8).tobytes()


def message_to_bits(message: str, msb_first: bool) -> np.ndarray:
    payload = message.encode("utf-8") + b"\x00"
    bits = []
    for byte in payload:
        if msb_first:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        else:
            for i in range(8):
                bits.append((byte >> i) & 1)
    return np.asarray(bits, dtype=np.uint8)


def plant_bits(image: np.ndarray, bits: np.ndarray, layout: LsbLayout) -> np.ndarray:
    out = np.ascontiguousarray(_as_rgb(image).copy())
    pix = out.reshape(-1, out.shape[2])
    bit = int(layout["bit"])
    walk = layout["walk"]
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    clear = np.uint8(0xFF ^ (1 << bit))
    n_pix = pix.shape[0]

    def write_channel(idx: int, payload: np.ndarray) -> None:
        n = min(n_pix, payload.size)
        if n == 0:
            return
        col = pix[:n, idx]
        pix[:n, idx] = (col & clear) | (payload[:n] << bit)

    if walk in _WALK_INDEX:
        if bits.size > n_pix:
            raise ValueError("message does not fit in this layout")
        write_channel(_WALK_INDEX[walk], bits)
        return out
    if bits.size > n_pix * 3:
        raise ValueError("message does not fit in this layout")
    if walk == "rgb":
        write_channel(0, bits[0::3])
        write_channel(1, bits[1::3])
        write_channel(2, bits[2::3])
        return out
    if walk == "rgb-planar":
        write_channel(0, bits[:n_pix])
        write_channel(1, bits[n_pix : n_pix * 2])
        write_channel(2, bits[n_pix * 2 : n_pix * 3])
        return out
    raise ValueError(f"unknown walk {walk}")
