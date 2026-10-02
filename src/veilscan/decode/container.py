"""Keyless container text: PNG tEXt/zTXt/iTXt, JPEG COM, EXIF comments."""

from __future__ import annotations

import struct
import zlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
from PIL.ExifTags import TAGS

PREFERRED_KEYS = {
    "comment",
    "description",
    "watermark",
    "message",
    "text",
    "usercomment",
    "imagedescription",
    "copyright",
    "xpcomment",
    "artist",
    "flag",
    "hint",
    "secret",
    "payload",
    "hidden",
    "note",
    "key",
}
SKIP_KEYS = {
    "software",
    "signature",
    "creation time",
    "interlace",
    "color type",
    "gamma",
    "chromaticity",
    "pixels per unit, x axis",
    "pixels per unit, y axis",
    "pixel units",
    "srgb",
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
LOSSY_NOTES = (
    "JPEG spatial LSB is not a v1 path. Comments and EXIF can still print. "
    "Spatial LSB usually dies after JPEG."
)


_MP3_SR = {
    3: (44100, 48000, 32000),
    2: (22050, 24000, 16000),
    0: (11025, 12000, 8000),
}
_MP3_BR = {
    (3, 3): (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 0),
    (3, 2): (0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 0),
    (3, 1): (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0),
    (2, 3): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0),
    (2, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    (2, 1): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    (0, 3): (0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256, 0),
    (0, 2): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
    (0, 1): (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
}


def _mp3_frame_len(data: bytes, off: int) -> int:
    if off + 4 > len(data) or data[off] != 0xFF or (data[off + 1] & 0xE0) != 0xE0:
        return 0
    ver = (data[off + 1] >> 3) & 0x3
    layer = (data[off + 1] >> 1) & 0x3
    bitrate_i = (data[off + 2] >> 4) & 0xF
    sr_i = (data[off + 2] >> 2) & 0x3
    if ver == 1 or layer == 0 or bitrate_i in (0, 15) or sr_i == 3:
        return 0
    br_table = _MP3_BR.get((ver, layer))
    sr_table = _MP3_SR.get(ver)
    if not br_table or not sr_table:
        return 0
    bitrate = br_table[bitrate_i] * 1000
    sample_rate = sr_table[sr_i]
    if bitrate <= 0 or sample_rate <= 0:
        return 0
    padding = (data[off + 2] >> 1) & 1
    if layer == 3:
        return int((12 * bitrate / sample_rate) + padding) * 4
    scale = 144 if ver == 3 else 72
    return int((scale * bitrate / sample_rate) + padding)


def looks_like_mp3(data: bytes) -> bool:
    if len(data) >= 10 and data[:3] == b"ID3" and data[3] in (2, 3, 4) and data[4] == 0:
        if any(b & 0x80 for b in data[6:10]):
            return False
        return True
    span = _mp3_frame_len(data, 0)
    if span < 4 or span + 4 > len(data):
        return False
    return _mp3_frame_len(data, span) >= 4


def sniff_kind(data: bytes) -> str:
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] in (b"II*\x00", b"MM\x00*") or data[:4] == b"II+\x00":
        return "tiff"
    if data.startswith(b"fLaC"):
        return "flac"
    if looks_like_mp3(data):
        return "mp3"
    return "unknown"


def lsb_safe_kind(kind: str) -> bool:
    return kind in {"png", "bmp", "tiff"}


def load_raw_rgb(data: bytes) -> tuple[np.ndarray | None, str, list[str]]:
    """Return HxWxC uint8 without flattening alpha onto white."""
    notes: list[str] = []
    kind = sniff_kind(data)
    try:
        im = Image.open(BytesIO(data))
        im = ImageOps.exif_transpose(im) or im
    except Exception as exc:  # noqa: BLE001
        return None, kind, [f"pixel load failed: {exc}"]
    if im.mode == "P":
        notes.append("Paletted image: expanded to RGBA. LSB of the file palette is not this array.")
        im = im.convert("RGBA")
    elif im.mode == "LA":
        im = im.convert("RGBA")
    elif im.mode == "L":
        im = im.convert("RGB")
    elif im.mode == "CMYK":
        notes.append("CMYK converted to RGB. LSB of the file is not this array.")
        im = im.convert("RGB")
    elif im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGB")
        notes.append(f"Converted mode to RGB for LSB walk.")
    arr = np.array(im)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr, kind, notes


def container_candidates(data: bytes) -> list[tuple[str, str, str]]:
    """(layout_id, key_or_marker, text)."""
    kind = sniff_kind(data)
    found: list[tuple[str, str, str]] = []
    if kind == "png":
        found.extend(_png_text(data))
    if kind == "jpeg":
        found.extend(_jpeg_com(data))
    found.extend(_exif_text(data))
    return found


def _png_text(data: bytes) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if not data.startswith(PNG_MAGIC):
        return out
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        start = pos + 8
        end = start + length
        if end + 4 > n or length < 0:
            break
        chunk = data[start:end]
        pos = end + 4
        if ctype == b"IEND":
            break
        if ctype == b"tEXt":
            key, _, val = chunk.partition(b"\x00")
            text = _latin_or_utf8(val)
            label = key.decode("latin-1", "replace")
            if text and _keep_key(label):
                out.append((f"png-tEXt:{label}", label, text))
        elif ctype == b"zTXt":
            key, _, rest = chunk.partition(b"\x00")
            if not rest:
                continue
            method = rest[0]
            payload = rest[1:]
            if method != 0:
                continue
            try:
                val = zlib.decompress(payload)
            except zlib.error:
                continue
            text = _latin_or_utf8(val)
            label = key.decode("latin-1", "replace")
            if text and _keep_key(label):
                out.append((f"png-zTXt:{label}", label, text))
        elif ctype == b"iTXt":
            parsed = _parse_itxt(chunk)
            if parsed:
                label, text = parsed
                if text and _keep_key(label):
                    out.append((f"png-iTXt:{label}", label, text))
    return out


def _parse_itxt(chunk: bytes) -> tuple[str, str] | None:
    try:
        key, rest = chunk.split(b"\x00", 1)
        if len(rest) < 2:
            return None
        flag = rest[0]
        method = rest[1]
        rest = rest[2:]
        lang, rest = rest.split(b"\x00", 1)
        trans, payload = rest.split(b"\x00", 1)
        if flag == 1:
            if method != 0:
                return None
            payload = zlib.decompress(payload)
        text = payload.decode("utf-8", "replace").strip()
        label = key.decode("latin-1", "replace")
        return label, text
    except (ValueError, zlib.error, UnicodeDecodeError):
        return None


def _jpeg_com(data: bytes) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if not data.startswith(JPEG_MAGIC):
        return out
    i = 2
    n = len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1
        if marker in (0x00, 0x01, 0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
            continue
        if i + 2 > n:
            break
        seglen = struct.unpack(">H", data[i : i + 2])[0]
        if seglen < 2 or i + seglen > n:
            break
        payload = data[i + 2 : i + seglen]
        i += seglen
        if marker == 0xFE:
            text = _latin_or_utf8(payload).strip("\x00").strip()
            if text:
                out.append(("jpeg-COM", "COM", text))
        if marker == 0xDA:
            break
    return out


def _exif_text(data: bytes) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    try:
        im = Image.open(BytesIO(data))
    except Exception:
        return out
    try:
        exif = im.getexif()
    except Exception:
        return out
    if not exif:
        return out
    for tag_id, value in exif.items():
        name = TAGS.get(tag_id, str(tag_id))
        if not _keep_key(name):
            continue
        text = _exif_value_to_text(value)
        if text:
            out.append((f"exif:{name}", name, text))
    try:
        ifd = exif.get_ifd(0x8769)  # Exif IFD
    except Exception:
        ifd = None
    if ifd:
        for tag_id, value in ifd.items():
            name = TAGS.get(tag_id, str(tag_id))
            if not _keep_key(name):
                continue
            text = _exif_value_to_text(value)
            if text:
                out.append((f"exif:{name}", name, text))
    return out


def _exif_value_to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        if value.startswith(b"ASCII\x00\x00\x00"):
            value = value[8:]
        elif value.startswith(b"UNICODE\x00"):
            try:
                return value[8:].decode("utf-16", "replace").strip("\x00").strip()
            except UnicodeDecodeError:
                return ""
        return _latin_or_utf8(value).strip("\x00").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _latin_or_utf8(raw: bytes) -> str:
    if not raw:
        return ""
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("latin-1", "replace").strip()


def _keep_key(name: str) -> bool:
    key = name.strip().lower()
    if key in SKIP_KEYS:
        return False
    if key in PREFERRED_KEYS:
        return True
    # Unknown keys: keep if they look like a human comment slot.
    if any(part in key for part in ("comment", "descript", "message", "watermark", "title", "flag", "secret", "hint")):
        return True
    return False


def plant_png_text(rgb: np.ndarray, message: str, key: str = "Comment") -> bytes:
    from PIL.PngImagePlugin import PngInfo

    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    info = PngInfo()
    info.add_text(key, message)
    buf = BytesIO()
    im.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def plant_png_ztxt(rgb: np.ndarray, message: str, key: str = "Comment") -> bytes:
    """Eval-only compressed PNG text chunk."""
    from PIL.PngImagePlugin import PngInfo

    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    info = PngInfo()
    info.add_text(key, message, zip=True)
    buf = BytesIO()
    im.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def plant_png_itxt(
    rgb: np.ndarray,
    message: str,
    key: str = "Description",
    lang: str = "en",
) -> bytes:
    """Eval-only iTXt chunk."""
    from PIL.PngImagePlugin import PngInfo

    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    info = PngInfo()
    info.add_itxt(key, message, lang=lang, tkey=key)
    buf = BytesIO()
    im.save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()


def plant_jpeg_com(rgb: np.ndarray, message: str, quality: int = 92) -> bytes:
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality, comment=message.encode("utf-8"))
    return buf.getvalue()
