"""Inspect packed bitstreams for flags, zlib, file magics, known stego headers."""

from __future__ import annotations

import zlib
from typing import Iterator

from veilscan.hunt.carve import GZIP, PDF_MAGIC, PNG_MAGIC, SEVENZ, ZIP_LOCAL
from veilscan.hunt.flags import find_flags, find_flags_bytes
from veilscan.hunt.strings import extract_strings

_MAGICS: list[tuple[bytes, str]] = [
    (PNG_MAGIC, "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (ZIP_LOCAL, "zip"),
    (PDF_MAGIC, "pdf"),
    (GZIP, "gzip"),
    (SEVENZ, "7z"),
    (b"GIF89a", "gif"),
    (b"GIF87a", "gif"),
]

_ZLIB_HDRS = (b"\x78\x01", b"\x78\x9c", b"\x78\xda")


def _try_inflate(blob: bytes) -> bytes | None:
    if not blob or len(blob) < 8:
        return None
    for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS, zlib.MAX_WBITS | 16):
        try:
            out = zlib.decompress(blob, wbits)
        except zlib.error:
            continue
        if out:
            return out
    return None


def inspect_payload(raw: bytes, cre) -> Iterator[dict]:
    if not raw:
        return
    sample = raw[: 256 * 1024]
    flags = find_flags_bytes(sample, cre)
    if flags:
        try:
            text = sample.decode("utf-8", "replace")
        except Exception:
            text = sample.decode("latin-1", "replace")
        yield {"kind": "flags", "flags": flags, "text": text[:4096], "confidence": 0.96}

    for s in extract_strings(sample, min_len=8, limit=80):
        hits = find_flags(s, cre)
        if hits:
            yield {"kind": "strings", "flags": hits, "text": s[:4096], "confidence": 0.93}

    for mag, name in _MAGICS:
        idx = sample.find(mag)
        if idx < 0:
            continue
        blob = sample[idx:]
        if name == "zip":
            import io
            import zipfile

            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    if not zf.infolist():
                        continue
            except Exception:
                continue
        elif name == "gzip":
            if _try_inflate(blob[: 256 * 1024]) is None:
                continue
        elif name == "png":
            if len(blob) < 16 or blob[12:16] != b"IHDR":
                continue
        elif name in {"jpeg", "gif", "7z", "pdf"}:
            if idx != 0:
                continue
        else:
            if idx != 0:
                continue
        yield {
            "kind": "magic",
            "name": name,
            "offset": idx,
            "text": f"{name} signature at bitstream offset {idx}",
            "confidence": 0.9,
            "flags": find_flags_bytes(blob[:4096], cre),
        }

    up = sample.upper()
    if b"OPENSTEGO" in up:
        yield {"kind": "openstego", "text": "OPENSTEGO header in bitstream", "confidence": 0.88, "flags": []}
    if b"CAMOUFLAGE" in up:
        yield {"kind": "camouflage", "text": "CAMOUFLAGE header in bitstream", "confidence": 0.88, "flags": []}

    for hdr in _ZLIB_HDRS + (GZIP,):
        start = 0
        found = 0
        while found < 3:
            idx = sample.find(hdr, start)
            if idx < 0:
                break
            start = idx + 1
            if idx > 64 and hdr == GZIP:
                continue
            inflated = _try_inflate(sample[idx : idx + 512 * 1024])
            if not inflated:
                continue
            found += 1
            hits = find_flags_bytes(inflated, cre)
            preview = ""
            try:
                preview = inflated[:512].decode("utf-8")
            except UnicodeDecodeError:
                preview = inflated[:512].decode("latin-1", "replace")
            yield {
                "kind": "zlib",
                "flags": hits,
                "text": (preview if hits or preview.isprintable() else f"inflated {len(inflated)} bytes"),
                "confidence": 0.94 if hits else 0.7,
                "offset": idx,
            }
