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
from veilscan.decode.embed_text import embed_text_array, embed_text_png_bytes
from veilscan.decode.lsb import message_to_bits
from veilscan.hunt.audio import plant_wav_glyphs, plant_wav_lsb, plant_wav_qr, plant_wav_tone
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


def _plant_gif_comment(flag: str) -> bytes:
    buf = io.BytesIO()
    Image.new("P", (2, 2), 1).save(buf, format="GIF")
    raw = buf.getvalue()
    if not raw.endswith(b"\x3b"):
        raw = raw + b"\x3b"
    payload = flag.encode("utf-8")
    blocks = bytearray()
    while payload:
        chunk = payload[:255]
        payload = payload[255:]
        blocks.append(len(chunk))
        blocks.extend(chunk)
    blocks.append(0)
    ext = b"\x21\xfe" + bytes(blocks)
    return raw[:-1] + ext + b"\x3b"


def _plant_webp_xmp(flag: str) -> bytes:
    payload = flag.encode("utf-8")
    chunk = b"XMP " + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        chunk += b"\x00"
    body = b"WEBP" + chunk
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _plant_column_r(rgb: np.ndarray, flag: str) -> bytes:
    bits = message_to_bits(flag, msb_first=True)
    out = np.ascontiguousarray(rgb.copy())
    ch = out[:, :, 0]
    flat = ch.ravel(order="F").copy()
    n = min(flat.size, bits.size)
    flat[:n] = (flat[:n] & np.uint8(0xFE)) | bits[:n]
    ch[:, :] = flat.reshape(ch.shape, order="F")
    return _png_bytes(out)


def _plant_zip_png_ztxt(rgb: np.ndarray, flag: str) -> bytes:
    hidden = plant_png_ztxt(rgb, flag, key="Comment")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("hidden.png", hidden)
    return _png_bytes(rgb) + buf.getvalue()


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

    flag_gif = "FLAG{gif-comment}"
    p = out_dir / "gif-comment.gif"
    p.write_bytes(_plant_gif_comment(flag_gif))
    cases.append(GymCase("gif-comment", p, flag_gif))

    flag_webp = "FLAG{webp-xmp}"
    p = out_dir / "webp-xmp.webp"
    p.write_bytes(_plant_webp_xmp(flag_webp))
    cases.append(GymCase("webp-xmp", p, flag_webp))

    flag_bmp = "FLAG{bmp-r-bit0}"
    p = out_dir / "bmp-r-bit0.bmp"
    marked = embed_text_array(_cover(50, 80), flag_bmp, layout_id="r-bit0-msb")
    buf = io.BytesIO()
    Image.fromarray(marked).save(buf, format="BMP")
    p.write_bytes(buf.getvalue())
    cases.append(GymCase("bmp-r-bit0", p, flag_bmp))

    flag_col = "FLAG{zsteg-col-bit0}"
    p = out_dir / "zsteg-col-bit0.png"
    p.write_bytes(_plant_column_r(_cover(51, 80), flag_col))
    cases.append(GymCase("zsteg-col-bit0", p, flag_col))

    flag_w16 = "FLAG{wav-pcm16-lsb}"
    p = out_dir / "wav-pcm16-lsb.wav"
    p.write_bytes(plant_wav_lsb(flag_w16, bits=16, channels=1, channel=0))
    cases.append(GymCase("wav-pcm16-lsb", p, flag_w16))

    flag_w8 = "FLAG{wav-pcm8-lsb}"
    p = out_dir / "wav-pcm8-lsb.wav"
    p.write_bytes(plant_wav_lsb(flag_w8, bits=8, channels=1, channel=0))
    cases.append(GymCase("wav-pcm8-lsb", p, flag_w8))

    flag_right = "FLAG{wav-stereo-right}"
    p = out_dir / "wav-stereo-right.wav"
    p.write_bytes(plant_wav_lsb(flag_right, bits=16, channels=2, channel=1))
    cases.append(GymCase("wav-stereo-right", p, flag_right))

    flag_rev = "FLAG{wav-lsb-reversed}"
    p = out_dir / "wav-lsb-reversed.wav"
    p.write_bytes(plant_wav_lsb(flag_rev, bits=16, channels=1, channel=0, reverse=True))
    cases.append(GymCase("wav-lsb-reversed", p, flag_rev))

    flag_tone = "FLAG{wav-tone-bytes}"
    p = out_dir / "wav-spectrogram-tone.wav"
    p.write_bytes(plant_wav_tone(flag_tone))
    cases.append(GymCase("wav-spectrogram-tone", p, flag_tone))

    flag_qr = "FLAG{wav-spectrogram-qr}"
    p = out_dir / "wav-spectrogram-qr.wav"
    qr_bytes = plant_wav_qr(flag_qr)
    if qr_bytes:
        p.write_bytes(qr_bytes)
        cases.append(GymCase("wav-spectrogram-qr", p, flag_qr))
    else:
        cases.append(GymCase("wav-spectrogram-qr", p, flag_qr, required=False, skipped="opencv QR encode failed"))

    flag_nest = "FLAG{zip-png-ztxt}"
    p = out_dir / "zip-png-ztxt.png"
    p.write_bytes(_plant_zip_png_ztxt(_cover(52), flag_nest))
    cases.append(GymCase("zip-png-ztxt", p, flag_nest))

    from veilscan.hunt.pdf import (
        plant_pdf_comment,
        plant_pdf_embed,
        plant_pdf_incremental,
        plant_pdf_info,
        plant_pdf_invisible,
        plant_pdf_js,
        plant_pdf_nested_png,
        plant_pdf_trail,
    )

    flag_pdf_info = "FLAG{pdf-info}"
    p = out_dir / "pdf-info.pdf"
    p.write_bytes(plant_pdf_info(flag_pdf_info))
    cases.append(GymCase("pdf-info", p, flag_pdf_info))

    flag_pdf_js = "FLAG{pdf-js}"
    p = out_dir / "pdf-js.pdf"
    p.write_bytes(plant_pdf_js(flag_pdf_js))
    cases.append(GymCase("pdf-js", p, flag_pdf_js))

    flag_pdf_embed = "FLAG{pdf-embed}"
    p = out_dir / "pdf-embed.pdf"
    p.write_bytes(plant_pdf_embed(flag_pdf_embed))
    cases.append(GymCase("pdf-embed", p, flag_pdf_embed))

    flag_pdf_trail = "FLAG{pdf-trail}"
    p = out_dir / "pdf-trail.pdf"
    p.write_bytes(plant_pdf_trail(flag_pdf_trail))
    cases.append(GymCase("pdf-trail", p, flag_pdf_trail))

    flag_pdf_comment = "FLAG{pdf-comment}"
    p = out_dir / "pdf-comment.pdf"
    p.write_bytes(plant_pdf_comment(flag_pdf_comment))
    cases.append(GymCase("pdf-comment", p, flag_pdf_comment))

    flag_pdf_invisible = "FLAG{pdf-invisible}"
    p = out_dir / "pdf-invisible.pdf"
    p.write_bytes(plant_pdf_invisible(flag_pdf_invisible))
    cases.append(GymCase("pdf-invisible", p, flag_pdf_invisible))

    flag_pdf_old = "FLAG{pdf-old}"
    p = out_dir / "pdf-incremental.pdf"
    p.write_bytes(plant_pdf_incremental(flag_pdf_old))
    cases.append(GymCase("pdf-incremental", p, flag_pdf_old))

    flag_pdf_png = "FLAG{pdf-nested-png}"
    p = out_dir / "pdf-nested-png.pdf"
    nested_png = plant_png_ztxt(_cover(60), flag_pdf_png, key="Comment")
    p.write_bytes(plant_pdf_nested_png(nested_png))
    cases.append(GymCase("pdf-nested-png", p, flag_pdf_png))

    import shutil

    from veilscan.hunt.compressed_audio import plant_flac_comment, plant_flac_lsb, plant_mp3_id3

    flag_id3 = "FLAG{mp3-id3}"
    p = out_dir / "mp3-id3.mp3"
    p.write_bytes(plant_mp3_id3(flag_id3))
    cases.append(GymCase("mp3-id3", p, flag_id3))

    flag_fc = "FLAG{flac-comment}"
    p = out_dir / "flac-comment.flac"
    p.write_bytes(plant_flac_comment(flag_fc))
    cases.append(GymCase("flac-comment", p, flag_fc))

    flag_flsb = "FLAG{flac-lsb}"
    p = out_dir / "flac-lsb.flac"
    try:
        import miniaudio  # noqa: F401
    except ImportError:
        cases.append(GymCase("flac-lsb", p, flag_flsb, required=False, skipped="miniaudio missing"))
    else:
        p.write_bytes(plant_flac_lsb(flag_flsb))
        cases.append(GymCase("flac-lsb", p, flag_flsb))

    flag_ink = "FLAG{SPEC-INK}"
    p = out_dir / "wav-spectrogram-ocr.wav"
    p.write_bytes(plant_wav_glyphs(flag_ink))
    if shutil.which("tesseract"):
        cases.append(GymCase("wav-spectrogram-ocr", p, flag_ink))
    else:
        cases.append(
            GymCase(
                "wav-spectrogram-ocr",
                p,
                flag_ink,
                required=False,
                skipped="tesseract not on PATH",
            )
        )

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
