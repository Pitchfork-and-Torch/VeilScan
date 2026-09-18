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

from veilscan.decode.container import (
    plant_jpeg_com,
    plant_png_itxt,
    plant_png_text,
    plant_png_ztxt,
)
from veilscan.decode.embed_text import embed_text_png_bytes
from veilscan.decode.lsb import message_to_bits
from veilscan.generators import synthetic_cover
from veilscan.hunt.pipeline import hunt_path


@dataclass
class GymCase:
    name: str
    path: Path
    flag: str
    required: bool = True
    skipped: str | None = None
    wordlist: Path | None = None


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


def _plant_plane(plane: np.ndarray, flag: str, bit: int = 0) -> np.ndarray:
    bits = message_to_bits(flag, msb_first=True)
    out = np.ascontiguousarray(plane.copy())
    flat = out.reshape(-1)
    n = min(flat.size, bits.size)
    clear = np.uint8(0xFF ^ (1 << bit))
    flat[:n] = (flat[:n] & clear) | (bits[:n] << bit)
    return out


def plant_alpha_lsb(rgb: np.ndarray, flag: str) -> bytes:
    alpha = _plant_plane(np.full(rgb.shape[:2], 200, dtype=np.uint8), flag, bit=0)
    rgba = np.dstack([rgb, alpha])
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def plant_palette_lsb(flag: str, size: int = 64, seed: int = 41) -> bytes:
    rng = np.random.default_rng(seed)
    idx = _plant_plane(rng.integers(0, 256, (size, size), dtype=np.uint8), flag, bit=0)
    pal = []
    for i in range(256):
        pal.extend((i, i, i))
    im = Image.fromarray(idx, mode="P")
    im.putpalette(pal)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def plant_jsteg(rgb: np.ndarray, flag: str, quality: int = 90) -> bytes:
    import tempfile

    import jpeglib

    from veilscan.decode.jsteg import ZZ

    bits = message_to_bits(flag, msb_first=True)
    tmp_in = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp_in.close()
    tmp_out.close()
    try:
        Image.fromarray(rgb).save(tmp_in.name, format="JPEG", quality=quality)
        im = jpeglib.read_dct(tmp_in.name)
        y = np.array(im.Y, copy=True)
        bi = 0
        nbr, nbc = y.shape[:2]
        for by in range(nbr):
            for bx in range(nbc):
                flat = y[by, bx].reshape(64).copy()
                for idx in ZZ:
                    if int(idx) == 0:
                        continue
                    v = int(flat[idx])
                    if v == 0:
                        continue
                    if bi >= bits.size:
                        break
                    nv = (v & ~1) | int(bits[bi])
                    if nv == 0:
                        nv = 2
                    flat[idx] = nv
                    bi += 1
                y[by, bx] = flat.reshape(8, 8)
                if bi >= bits.size:
                    break
            if bi >= bits.size:
                break
        if bi < bits.size:
            raise ValueError(f"jsteg plant did not fit ({bi}/{bits.size})")
        im.Y = y
        im.write_dct(tmp_out.name)
        return Path(tmp_out.name).read_bytes()
    finally:
        Path(tmp_in.name).unlink(missing_ok=True)
        Path(tmp_out.name).unlink(missing_ok=True)


def plant_bitplane_qr(rgb: np.ndarray, flag: str) -> bytes:
    import cv2

    enc = cv2.QRCodeEncoder.create()
    qr = enc.encode(flag)
    qr_bin = (np.asarray(qr) > 0).astype(np.uint8)
    h, w = rgb.shape[:2]
    big = cv2.resize(qr_bin, (w, h), interpolation=cv2.INTER_NEAREST)
    out = np.ascontiguousarray(rgb.copy())
    out[:, :, 0] = (out[:, :, 0] & np.uint8(0xFE)) | big
    buf = io.BytesIO()
    Image.fromarray(out).save(buf, format="PNG")
    return buf.getvalue()


def write_gym(out_dir: Path) -> list[GymCase]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[GymCase] = []

    flag_text = "FLAG{png-text-comment}"
    p = out_dir / "png-tEXt-flag.png"
    p.write_bytes(plant_png_text(_cover(31), flag_text, key="flag"))
    cases.append(GymCase("png-tEXt-flag", p, flag_text))

    flag_ztxt = "FLAG{png-ztxt-comment}"
    p = out_dir / "png-zTXt-flag.png"
    p.write_bytes(plant_png_ztxt(_cover(40), flag_ztxt, key="Comment"))
    cases.append(GymCase("png-zTXt-flag", p, flag_ztxt))

    flag_itxt = "FLAG{png-itxt-comment}"
    p = out_dir / "png-iTXt-flag.png"
    p.write_bytes(plant_png_itxt(_cover(43), flag_itxt, key="Description"))
    cases.append(GymCase("png-iTXt-flag", p, flag_itxt))

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

    flag_rgb = "FLAG{zsteg-rgb-bit0}"
    p = out_dir / "zsteg-rgb-bit0.png"
    p.write_bytes(embed_text_png_bytes(_cover(35, 80), flag_rgb, layout_id="rgb-bit0-msb"))
    cases.append(GymCase("zsteg-rgb-bit0", p, flag_rgb))

    flag_a = "FLAG{alpha-lsb}"
    p = out_dir / "alpha-lsb.png"
    p.write_bytes(plant_alpha_lsb(_cover(36, 80), flag_a))
    cases.append(GymCase("alpha-lsb", p, flag_a))

    flag_pal = "FLAG{palette-lsb}"
    p = out_dir / "palette-lsb.png"
    p.write_bytes(plant_palette_lsb(flag_pal, size=80, seed=41))
    cases.append(GymCase("palette-lsb", p, flag_pal))

    flag_qr = "FLAG{bitplane-qr}"
    p = out_dir / "bitplane-qr.png"
    p.write_bytes(plant_bitplane_qr(_cover(37, 128), flag_qr))
    cases.append(GymCase("bitplane-qr", p, flag_qr))

    from veilscan.decode.jsteg import available as jsteg_ok

    flag_js = "FLAG{jsteg-ac}"
    p = out_dir / "jsteg-flag.jpg"
    if jsteg_ok():
        p.write_bytes(plant_jsteg(_cover(38, 128), flag_js))
        cases.append(GymCase("jsteg-flag", p, flag_js))
    else:
        cases.append(GymCase("jsteg-flag", p, flag_js, required=False, skipped="jpeglib missing"))

    flag_sh = "FLAG{steghide-password}"
    p = out_dir / "steghide-password.jpg"
    cover_jpg = out_dir / "_steghide-cover.jpg"
    Image.fromarray(_cover(39, 160)).save(cover_jpg, format="JPEG", quality=90)
    from veilscan.hunt.adapters import plant_steghide

    wl = out_dir / "gym-wordlist.txt"
    wl.write_text("wrongpass\nveilscan-gym\n", encoding="utf-8")
    if plant_steghide(cover_jpg, (flag_sh + "\n").encode("utf-8"), "veilscan-gym", p):
        cases.append(GymCase("steghide-password", p, flag_sh, required=False, wordlist=wl))
    else:
        cases.append(
            GymCase(
                "steghide-password",
                p,
                flag_sh,
                required=False,
                skipped="steghide not on PATH",
                wordlist=wl,
            )
        )

    return cases


def run_gym(out_dir: Path) -> dict:
    cases = write_gym(out_dir)
    rows = []
    ok = 0
    attempted = 0
    for case in cases:
        if case.skipped:
            rows.append(
                {
                    "name": case.name,
                    "flag": case.flag,
                    "hit": False,
                    "skipped": case.skipped,
                    "flags": [],
                    "elapsed_ms": 0.0,
                    "required": case.required,
                }
            )
            continue
        attempted += 1
        hunt_out = out_dir / f"out-{case.name}"
        result = hunt_path(case.path, out_dir=hunt_out, wordlist=case.wordlist)
        hit = case.flag in result.flags or any(case.flag in (f.text or "") for f in result.findings)
        if hit:
            ok += 1
        rows.append(
            {
                "name": case.name,
                "flag": case.flag,
                "hit": hit,
                "skipped": None,
                "flags": result.flags,
                "elapsed_ms": result.elapsed_ms,
                "required": case.required,
            }
        )
    required_miss = [
        row["name"]
        for row in rows
        if row["required"] and not row.get("skipped") and not row["hit"]
    ]
    req = [row for row in rows if row["required"] and not row.get("skipped")]
    req_hits = sum(1 for row in req if row["hit"])
    return {
        "n": len(req),
        "hits": req_hits,
        "tpr": (req_hits / len(req)) if req else 1.0,
        "optional_hits": ok - req_hits,
        "required_miss": required_miss,
        "cases": rows,
    }
