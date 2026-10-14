"""JSteg-style LSB of JPEG quantized DCT AC coefficients. Optional jpeglib."""

from __future__ import annotations

import os
from typing import Iterator

import numpy as np

from veilscan.decode.lsb import pack_bits

ZZ = np.array(
    [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
    ],
    dtype=np.int32,
)


def available() -> bool:
    try:
        import jpeglib  # noqa: F401
    except Exception:
        return False
    return True


def _ac_bits(blocks: np.ndarray, skip_zero: bool = True) -> np.ndarray:
    nbr, nbc = blocks.shape[:2]
    bits: list[int] = []
    for by in range(nbr):
        for bx in range(nbc):
            flat = blocks[by, bx].reshape(64)
            for idx in ZZ:
                if idx == 0:
                    continue
                v = int(flat[idx])
                if skip_zero and v == 0:
                    continue
                bits.append(v & 1)
    return np.asarray(bits, dtype=np.uint8)


def iter_jsteg_payloads(path: str | bytes, box: tuple[int, int, int, int] | None = None) -> Iterator[tuple[str, bytes]]:
    """Yield (layout_id, packed_bytes) for classic JSteg walks."""
    try:
        import jpeglib
    except Exception:
        return
    tmp_name = None
    try:
        if isinstance(path, (bytes, bytearray)):
            import tempfile
            from pathlib import Path

            fd, tmp_name = tempfile.mkstemp(suffix=".jpg")
            os.close(fd)
            Path(tmp_name).write_bytes(bytes(path))
            im = jpeglib.read_dct(tmp_name)
        else:
            im = jpeglib.read_dct(str(path))
    except Exception:
        return
    finally:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
    Y = im.Y
    bh, bw = Y.shape[:2]
    if box is not None:
        x, y, w, h = box
        bx0 = max(0, x // 8)
        by0 = max(0, y // 8)
        bx1 = min(bw, (x + w + 7) // 8)
        by1 = min(bh, (y + h + 7) // 8)
        region = Y[by0:by1, bx0:bx1]
        tag = f"jsteg-Y-{x},{y}"
    else:
        region = Y
        tag = "jsteg-Y"
    for skip_zero in (True, False):
        bits = _ac_bits(region, skip_zero=skip_zero)
        for msb in (True, False):
            raw = pack_bits(bits, msb_first=msb)
            z = "nz" if skip_zero else "allac"
            o = "msb" if msb else "lsb"
            yield f"{tag}-{z}-zz-{o}", raw
