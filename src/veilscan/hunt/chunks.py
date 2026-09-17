"""Container metadata without PREFERRED_KEYS. Hunt dumps CTF keys decode used to drop."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from io import BytesIO

from PIL import Image
from PIL.ExifTags import TAGS

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"


@dataclass
class ChunkHit:
    family: str
    method: str
    key: str
    text: str
    offset: int | None = None
    extra: str = ""


def _latin_or_utf8(raw: bytes) -> str:
    if not raw:
        return ""
    try:
        return raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        return raw.decode("latin-1", "replace").strip()


def _printable_payload(raw: bytes) -> str:
    if not raw:
        return ""
    text = _latin_or_utf8(raw)
    if not text:
        return ""
    ok = sum(1 for ch in text if ch.isprintable() or ch in "\t\n\r")
    if ok / max(len(text), 1) < 0.85:
        return ""
    return text


def png_chunks(data: bytes) -> list[ChunkHit]:
    out: list[ChunkHit] = []
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
        off = pos
        pos = end + 4
        try:
            label = ctype.decode("latin-1")
        except Exception:
            label = repr(ctype)
        if ctype == b"IEND":
            break
        if ctype == b"tEXt":
            key, _, val = chunk.partition(b"\x00")
            text = _latin_or_utf8(val)
            if text:
                out.append(ChunkHit("container", f"png-tEXt:{_latin_or_utf8(key)}", _latin_or_utf8(key), text, off))
        elif ctype == b"zTXt":
            key, _, rest = chunk.partition(b"\x00")
            if rest and rest[0] == 0:
                try:
                    val = zlib.decompress(rest[1:])
                except zlib.error:
                    val = b""
                text = _latin_or_utf8(val)
                if text:
                    out.append(ChunkHit("container", f"png-zTXt:{_latin_or_utf8(key)}", _latin_or_utf8(key), text, off))
        elif ctype == b"iTXt":
            parsed = _parse_itxt(chunk)
            if parsed:
                k, text = parsed
                out.append(ChunkHit("container", f"png-iTXt:{k}", k, text, off))
        elif ctype not in {
            b"IHDR",
            b"IDAT",
            b"PLTE",
            b"IEND",
            b"gAMA",
            b"cHRM",
            b"pHYs",
            b"iCCP",
            b"sRGB",
            b"bKGD",
            b"tIME",
            b"sBIT",
            b"tRNS",
        }:
            text = _printable_payload(chunk)
            if text:
                out.append(ChunkHit("container", f"png-chunk:{label}", label, text, off, extra="unknown-or-ancillary"))
            elif chunk:
                preview = chunk[:64].hex()
                out.append(
                    ChunkHit(
                        "container",
                        f"png-chunk:{label}",
                        label,
                        f"binary {len(chunk)} bytes hex={preview}",
                        off,
                        extra="binary",
                    )
                )
    return out


def _parse_itxt(chunk: bytes) -> tuple[str, str] | None:
    try:
        key, rest = chunk.split(b"\x00", 1)
        if len(rest) < 2:
            return None
        flag = rest[0]
        method = rest[1]
        rest = rest[2:]
        _lang, rest = rest.split(b"\x00", 1)
        _trans, payload = rest.split(b"\x00", 1)
        if flag == 1:
            if method != 0:
                return None
            payload = zlib.decompress(payload)
        text = payload.decode("utf-8", "replace").strip()
        label = key.decode("latin-1", "replace")
        if text:
            return label, text
    except (ValueError, zlib.error, UnicodeDecodeError):
        return None
    return None


def jpeg_markers(data: bytes) -> list[ChunkHit]:
    out: list[ChunkHit] = []
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
        off = i - 2
        i += seglen
        if marker == 0xFE:
            text = _latin_or_utf8(payload).strip("\x00").strip()
            if text:
                out.append(ChunkHit("container", "jpeg-COM", "COM", text, off))
        elif 0xE0 <= marker <= 0xEF:
            name = f"APP{marker - 0xE0}"
            text = _printable_payload(payload)
            if text:
                out.append(ChunkHit("container", f"jpeg-{name}", name, text, off))
            elif payload:
                out.append(
                    ChunkHit(
                        "container",
                        f"jpeg-{name}",
                        name,
                        f"binary {len(payload)} bytes",
                        off,
                        extra=payload[:16].hex(),
                    )
                )
        if marker == 0xDA:
            break
    out.extend(_exif_all(data))
    return out


def _exif_all(data: bytes) -> list[ChunkHit]:
    out: list[ChunkHit] = []
    try:
        im = Image.open(BytesIO(data))
        exif = im.getexif()
    except Exception:
        return out
    if not exif:
        return out

    def walk(mapping, prefix: str) -> None:
        for tag_id, value in mapping.items():
            name = TAGS.get(tag_id, str(tag_id))
            text = _exif_value_to_text(value)
            if text:
                out.append(ChunkHit("container", f"{prefix}:{name}", name, text))

    walk(exif, "exif")
    try:
        ifd = exif.get_ifd(0x8769)
    except Exception:
        ifd = None
    if ifd:
        walk(ifd, "exif")
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
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def gif_comments(data: bytes) -> list[ChunkHit]:
    out: list[ChunkHit] = []
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        return out
    i = 0
    n = len(data)
    while i + 2 < n:
        idx = data.find(b"\x21\xfe", i)
        if idx < 0:
            break
        i = idx + 2
        parts = bytearray()
        while i < n:
            blen = data[i]
            i += 1
            if blen == 0:
                break
            parts.extend(data[i : i + blen])
            i += blen
        text = _latin_or_utf8(bytes(parts))
        if text:
            out.append(ChunkHit("container", "gif-comment", "comment", text, idx))
    return out


def webp_chunks(data: bytes) -> list[ChunkHit]:
    out: list[ChunkHit] = []
    if not (data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP"):
        return out
    pos = 12
    n = len(data)
    while pos + 8 <= n:
        ctype = data[pos : pos + 4]
        length = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        start = pos + 8
        end = start + length
        if end > n:
            break
        payload = data[start:end]
        label = ctype.decode("latin-1", "replace")
        text = _printable_payload(payload)
        if text and ctype not in {b"VP8 ", b"VP8L", b"VP8X"}:
            out.append(ChunkHit("container", f"webp:{label}", label, text, pos))
        pos = end + (length & 1)
    return out


def extract_chunks(data: bytes) -> list[ChunkHit]:
    out: list[ChunkHit] = []
    out.extend(png_chunks(data))
    out.extend(jpeg_markers(data))
    out.extend(gif_comments(data))
    out.extend(webp_chunks(data))
    return out
