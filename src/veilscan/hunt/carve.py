"""Trailing bytes and embedded-file carve. Native. No binwalk required."""

from __future__ import annotations

import io
import struct
import zipfile
from dataclasses import dataclass

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8"
ZIP_LOCAL = b"PK\x03\x04"
PDF_MAGIC = b"%PDF"
RAR4 = b"Rar!\x1a\x07\x00"
RAR5 = b"Rar!\x1a\x07\x01\x00"
GZIP = b"\x1f\x8b\x08"
SEVENZ = b"7z\xbc\xaf'\x1c"
GIF87 = b"GIF87a"
GIF89 = b"GIF89a"
BMP = b"BM"
WEBP_RIFF = b"RIFF"

MAX_ARTIFACT = 8 * 1024 * 1024


@dataclass
class CarveHit:
    kind: str
    offset: int
    length: int
    payload: bytes
    note: str
    inner_files: list[tuple[str, bytes]]


def png_iend_end(data: bytes) -> int | None:
    if not data.startswith(PNG_MAGIC):
        return None
    pos = 8
    n = len(data)
    while pos + 8 <= n:
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        if length < 0 or pos + 12 + length > n:
            return None
        ctype = data[pos + 4 : pos + 8]
        pos = pos + 12 + length
        if ctype == b"IEND":
            return pos
    return None


def jpeg_eoi_end(data: bytes) -> int | None:
    if not data.startswith(JPEG_MAGIC):
        return None
    idx = data.rfind(b"\xff\xd9")
    if idx < 0:
        return None
    return idx + 2


def declared_end(data: bytes) -> tuple[str | None, int | None]:
    end = png_iend_end(data)
    if end is not None:
        return "png", end
    end = jpeg_eoi_end(data)
    if end is not None:
        return "jpeg", end
    if data.startswith(b"RIFF") and len(data) >= 8:
        riff_n = struct.unpack("<I", data[4:8])[0]
        return "riff", min(len(data), 8 + riff_n)
    return None, None


def _unzip(blob: bytes) -> list[tuple[str, bytes]]:
    out: list[tuple[str, bytes]] = []
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                if info.file_size > MAX_ARTIFACT:
                    continue
                try:
                    raw = zf.read(info)
                except Exception:
                    continue
                name = info.filename or "unnamed"
                out.append((name, raw))
    except Exception:
        return []
    return out


_MAGICS: list[tuple[bytes, str]] = [
    (PNG_MAGIC, "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (ZIP_LOCAL, "zip"),
    (PDF_MAGIC, "pdf"),
    (RAR5, "rar"),
    (RAR4, "rar"),
    (SEVENZ, "7z"),
    (GZIP, "gzip"),
    (GIF89, "gif"),
    (GIF87, "gif"),
]


def _skip_own_header(data: bytes, offset: int, kind: str) -> bool:
    if offset != 0:
        return False
    if kind == "png" and data.startswith(PNG_MAGIC):
        return True
    if kind == "jpeg" and data.startswith(JPEG_MAGIC):
        return True
    if kind == "gif" and data[:6] in (GIF87, GIF89):
        return True
    if kind == "pdf" and data.startswith(b"%PDF-"):
        return True
    return False


def _in_spans(offset: int, spans: list[tuple[int, int]] | None) -> bool:
    if not spans:
        return False
    return any(a <= offset < b for a, b in spans)


def carve(data: bytes, ignore_spans: list[tuple[int, int]] | None = None) -> list[CarveHit]:
    hits: list[CarveHit] = []
    n = len(data)
    kind, end = declared_end(data)
    if end is not None and end < n:
        trail = data[end:]
        inner = _unzip(trail)
        hits.append(
            CarveHit(
                kind="trailing",
                offset=end,
                length=n - end,
                payload=trail,
                note=f"{n - end} bytes after {kind or 'declared'} end (file is {n})",
                inner_files=inner,
            )
        )
        if inner:
            hits[-1].kind = "trailing-zip"

    seen_off: set[tuple[str, int]] = set()
    for magic, label in _MAGICS:
        start = 0
        while True:
            idx = data.find(magic, start)
            if idx < 0:
                break
            start = idx + 1
            if _skip_own_header(data, idx, label):
                continue
            if _in_spans(idx, ignore_spans):
                continue
            if (label, idx) in seen_off:
                continue
            seen_off.add((label, idx))
            blob = data[idx : min(n, idx + MAX_ARTIFACT)]
            inner: list[tuple[str, bytes]] = []
            if label == "zip":
                inner = _unzip(data[idx:])
                if inner:
                    blob = data[idx:]
            hits.append(
                CarveHit(
                    kind=label,
                    offset=idx,
                    length=len(blob),
                    payload=blob[: min(len(blob), MAX_ARTIFACT)],
                    note=f"{label} signature at offset {idx}",
                    inner_files=inner,
                )
            )

    # Dedup trailing zip vs zip-at-same-offset
    uniq: list[CarveHit] = []
    seen: set[tuple[str, int]] = set()
    for h in hits:
        key = (h.kind, h.offset)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq
