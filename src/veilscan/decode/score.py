"""UTF-8 / printable gates so photo LSB noise does not invent sentences."""

from __future__ import annotations

from veilscan.decode.layouts import MAX_MESSAGE_BYTES, MIN_FRAMED_CHARS, MIN_UNFRAMED_CHARS

_PRINTABLE_EXTRA = set("\t\n\r")


def _is_text_char(ch: str) -> bool:
    if ch in _PRINTABLE_EXTRA:
        return True
    return ch.isprintable() and ch not in "\x0b\x0c"


def utf8_text(data: bytes) -> str | None:
    if not data or len(data) > MAX_MESSAGE_BYTES:
        return None
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\ufeff" in text:
        text = text.replace("\ufeff", "")
    if not text:
        return None
    return text


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    ok = sum(1 for ch in text if _is_text_char(ch))
    return ok / len(text)


def score_text(text: str, framed: bool) -> float:
    n = len(text)
    min_n = MIN_FRAMED_CHARS if framed else MIN_UNFRAMED_CHARS
    if n < min_n:
        return 0.0
    pr = printable_ratio(text)
    need = 0.92 if framed else 0.99
    if pr < need:
        return 0.0
    if not framed and pr < 1.0:
        return 0.0
    length_term = min(n / 32.0, 1.0)
    s = 0.50 * pr + 0.30 * length_term + 0.20
    if framed:
        s = min(1.0, s + 0.15)
    return float(s)


def split_frames(raw: bytes) -> list[tuple[bytes, bool]]:
    """Return (payload, framed) candidates from packed LSB bytes."""
    out: list[tuple[bytes, bool]] = []
    if not raw:
        return out
    nul = raw.find(b"\x00")
    if nul >= 0:
        out.append((raw[:nul], True))
    if len(raw) >= 4:
        n = int.from_bytes(raw[:4], "little", signed=False)
        if 2 <= n <= min(MAX_MESSAGE_BYTES, len(raw) - 4):
            out.append((raw[4 : 4 + n], True))
    prefix = _printable_prefix_bytes(raw)
    if prefix:
        out.append((prefix, False))
    return out


def _printable_prefix_bytes(raw: bytes) -> bytes:
    acc = bytearray()
    i = 0
    while i < min(len(raw), MAX_MESSAGE_BYTES):
        b = raw[i]
        if 0x20 <= b <= 0x7E or b in (0x09, 0x0A, 0x0D):
            acc.append(b)
            i += 1
            continue
        if 0xC2 <= b <= 0xF4:
            break
        break
    return bytes(acc)
