"""Eval-only planted fixtures for hunt extract TPR. Not a hiding product."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from veilscan.decode.container import plant_jpeg_com, plant_png_text
from veilscan.generators import synthetic_cover
from veilscan.hunt.pipeline import hunt_path


@dataclass
class GymCase:
    name: str
    path: Path
    flag: str


def _cover(seed: int, size: int = 64) -> np.ndarray:
    return synthetic_cover(size, size, np.random.default_rng(seed), style="sine")


def _png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def _png_chunk(ctype: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def insert_png_chunk_before_iend(png: bytes, ctype: bytes, payload: bytes) -> bytes:
    pos = 8
    n = len(png)
    while pos + 8 <= n:
        length = struct.unpack(">I", png[pos : pos + 4])[0]
        ctype_here = png[pos + 4 : pos + 8]
        if ctype_here == b"IEND":
            return png[:pos] + _png_chunk(ctype, payload) + png[pos:]
        pos = pos + 12 + length
    raise ValueError("no IEND")


def plant_trailing_zip(rgb: np.ndarray, flag: str) -> bytes:
    png = _png_bytes(rgb)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("flag.txt", flag + "\n")
    return png + buf.getvalue()


def plant_unknown_chunk(rgb: np.ndarray, flag: str) -> bytes:
    png = _png_bytes(rgb)
    return insert_png_chunk_before_iend(png, b"faLg", flag.encode("utf-8"))


def write_gym(out_dir: Path) -> list[GymCase]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[GymCase] = []

    flag_text = "FLAG{png-text-comment}"
    p = out_dir / "png-tEXt-flag.png"
    p.write_bytes(plant_png_text(_cover(31), flag_text, key="flag"))
    cases.append(GymCase("png-tEXt-flag", p, flag_text))

    flag_com = "FLAG{jpeg-com}"
    p = out_dir / "jpeg-com-flag.jpg"
    p.write_bytes(plant_jpeg_com(_cover(32), flag_com))
    cases.append(GymCase("jpeg-com-flag", p, flag_com))

    flag_zip = "FLAG{trailing-zip}"
    p = out_dir / "trailing-zip.png"
    p.write_bytes(plant_trailing_zip(_cover(33), flag_zip))
    cases.append(GymCase("trailing-zip", p, flag_zip))

    flag_chunk = "FLAG{unknown-chunk}"
    p = out_dir / "png-unknown-chunk.png"
    p.write_bytes(plant_unknown_chunk(_cover(34), flag_chunk))
    cases.append(GymCase("png-unknown-chunk", p, flag_chunk))

    return cases


def run_gym(out_dir: Path) -> dict:
    cases = write_gym(out_dir)
    rows = []
    ok = 0
    for case in cases:
        hunt_out = out_dir / f"out-{case.name}"
        result = hunt_path(case.path, out_dir=hunt_out)
        hit = case.flag in result.flags or any(case.flag in (f.text or "") for f in result.findings)
        if hit:
            ok += 1
        rows.append(
            {
                "name": case.name,
                "flag": case.flag,
                "hit": hit,
                "flags": result.flags,
                "elapsed_ms": result.elapsed_ms,
            }
        )
    n = len(cases)
    return {
        "n": n,
        "hits": ok,
        "tpr": (ok / n) if n else 0.0,
        "cases": rows,
    }
