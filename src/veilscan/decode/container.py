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


def sniff_kind(data: bytes) -> str:
    if data.startswith(PNG_MAGIC):
        return "png"
    if data.startswith(JPEG_MAGIC):
        return "jpeg"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "webp"
    if data[:4] in (b"II*\x00", b"MM\x00*") or data[:4] == b"II+\x00":
        return "tiff"
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


def plant_jpeg_com(rgb: np.ndarray, message: str, quality: int = 92) -> bytes:
    im = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=quality, comment=message.encode("utf-8"))
    return buf.getvalue()
