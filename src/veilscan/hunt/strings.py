"""Printable strings from raw bytes (CTF strings -n 8)."""

from __future__ import annotations


def extract_strings(data: bytes, min_len: int = 8, limit: int = 400) -> list[str]:
    out: list[str] = []
    buf = bytearray()
    seen: set[str] = set()

    def flush() -> None:
        if len(buf) < min_len:
            buf.clear()
            return
        s = buf.decode("ascii")
        buf.clear()
        if s in seen:
            return
        seen.add(s)
        out.append(s)

    for b in data:
        if 0x20 <= b <= 0x7E:
            buf.append(b)
            if len(buf) > 4096:
                flush()
        else:
            flush()
            if len(out) >= limit:
                return out
    flush()
    return out[:limit]
