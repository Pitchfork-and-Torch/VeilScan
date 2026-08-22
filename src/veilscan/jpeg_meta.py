"""JPEG container inspect. No jpeglib required."""

from __future__ import annotations

from typing import Any

JPEG_SOI = b"\xff\xd8"

# Independent JPEG Group standard luminance table, raster 8x8.
_STD_LUMA = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

# Raster indices in zigzag visit order (DQT payload order).
_ZZ = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]


def is_jpeg(data: bytes | bytearray | memoryview | None) -> bool:
    if not data or len(data) < 3:
        return False
    return bytes(data[:2]) == JPEG_SOI


def inspect_jpeg(data: bytes | bytearray | memoryview | None) -> dict[str, Any]:
    """Parse SOI / DQT / SOF. Never raises on garbage."""
    out: dict[str, Any] = {
        "container": False,
        "quality_est": None,
        "subsampling": None,
        "components": None,
        "progressive": False,
    }
    if not is_jpeg(data):
        return out
    raw = bytes(data)
    out["container"] = True
    qtables: dict[int, list[int]] = {}
    i = 2
    n = len(raw)
    while i < n - 1:
        if raw[i] != 0xFF:
            i += 1
            continue
        while i < n and raw[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = raw[i]
        i += 1
        if marker in (0xD8, 0xD9):
            continue
        if 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0xDA:
            break
        if i + 2 > n:
            break
        seglen = int.from_bytes(raw[i : i + 2], "big")
        if seglen < 2:
            break
        payload = raw[i + 2 : i + seglen]
        i += seglen
        if marker == 0xDB:
            _read_dqt(payload, qtables)
        elif marker in (0xC0, 0xC1, 0xC2, 0xC3):
            out["progressive"] = marker == 0xC2
            comps, sub = _read_sof(payload)
            out["components"] = comps
            out["subsampling"] = sub
    luma = qtables.get(0) or (qtables[min(qtables)] if qtables else None)
    if luma and len(luma) == 64:
        out["quality_est"] = _quality_from_luma(luma)
    return out


def _read_dqt(payload: bytes, dest: dict[int, list[int]]) -> None:
    k = 0
    while k + 1 <= len(payload):
        info = payload[k]
        k += 1
        precision = info >> 4
        table_id = info & 0x0F
        step = 2 if precision else 1
        need = 64 * step
        if k + need > len(payload):
            return
        chunk = payload[k : k + need]
        k += need
        if step == 1:
            dest[table_id] = list(chunk)
        else:
            dest[table_id] = [
                int.from_bytes(chunk[j : j + 2], "big") for j in range(0, need, 2)
            ]


def _read_sof(payload: bytes) -> tuple[int | None, str | None]:
    if len(payload) < 6:
        return None, None
    ncomp = payload[5]
    if ncomp < 1 or len(payload) < 6 + 3 * ncomp:
        return ncomp, None
    samp: list[tuple[int, int]] = []
    for c in range(ncomp):
        byte = payload[6 + 3 * c + 1]
        samp.append((byte >> 4, byte & 0x0F))
    if not samp:
        return ncomp, None
    yh, yv = samp[0]
    if ncomp == 1:
        return ncomp, "4:4:4"
    cbh, cbv = samp[1] if len(samp) > 1 else (1, 1)
    if yh == 1 and yv == 1 and cbh == 1 and cbv == 1:
        label = "4:4:4"
    elif yh == 2 and yv == 1 and cbh == 1 and cbv == 1:
        label = "4:2:2"
    elif yh == 2 and yv == 2 and cbh == 1 and cbv == 1:
        label = "4:2:0"
    else:
        label = f"{yh}x{yv}:{cbh}x{cbv}"
    return ncomp, label


def _quality_from_luma(dqt_zz: list[int]) -> int:
    std_zz = [_STD_LUMA[idx] for idx in _ZZ]
    scales = []
    for q, s in zip(dqt_zz, std_zz):
        if s <= 0:
            continue
        scales.append(float(max(q, 1)) / float(s))
    if not scales:
        return 50
    scales.sort()
    mid = scales[len(scales) // 2]
    if mid <= 0.02:
        est = 100
    elif mid <= 1.0:
        est = 100.0 - 50.0 * mid
    else:
        est = 50.0 / mid
    return int(max(1, min(100, round(est))))


def jpeg_freq_weight(
    quality_est: int | None,
    *,
    jpeg_like: bool,
    blockiness: float = 0.0,
) -> float:
    """FSNet blend weight. 0 if not jpeg_like. Heavier at low Q."""
    if not jpeg_like:
        return 0.0
    if quality_est is not None:
        q = int(quality_est)
        if q <= 50:
            return 0.70
        if q >= 90:
            return 0.25
        t = (q - 50) / 40.0
        return float(round(0.70 - t * 0.45, 4))
    if blockiness >= 1.20:
        return 0.70
    if blockiness >= 1.10:
        return 0.50
    return 0.25
