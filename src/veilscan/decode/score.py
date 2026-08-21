"""UTF-8 / printable gates so photo LSB noise does not invent sentences."""

from __future__ import annotations

import re

from veilscan.decode.layouts import MAX_MESSAGE_BYTES, MIN_FRAMED_CHARS, MIN_UNFRAMED_CHARS

_TOKEN_RE = re.compile(r"^[A-Z]{2,}[A-Z0-9_]*:[A-Z0-9_.:/=+\-]{2,64}$")
_TOKEN_FIND = re.compile(r"[A-Z]{2,}[A-Z0-9_]*:[A-Z0-9_.:/=+\-]{2,64}")
_WEIRD = set(",'\"%+#$&*;<>[]{}\\|`~")

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


def looks_like_token(text: str) -> bool:
    return bool(_TOKEN_RE.match(text.strip()))


def looks_like_english(text: str) -> bool:
    t = text.strip()
    if sum(ch in _WEIRD for ch in t) >= 2:
        return False
    letters = sum(ch.isalpha() for ch in t)
    if letters / len(t) < 0.70:
        return False
    vowels = sum(ch.lower() in "aeiou" for ch in t)
    if letters == 0 or vowels / letters < 0.22:
        return False
    if vowels < 2:
        return False
    if " " not in t:
        return False
    return True


def looks_like_message(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    low = t.lower()
    if low.startswith(("http://", "https://", "ftp://", "www.")):
        return True
    if looks_like_token(t):
        return True
    return looks_like_english(t)


def score_text(text: str, framed: bool) -> float:
    n = len(text)
    min_n = MIN_FRAMED_CHARS if framed else MIN_UNFRAMED_CHARS
    if n < min_n:
        return 0.0
    if not looks_like_message(text):
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
    if looks_like_token(text):
        s = min(1.0, s + 0.05)
    return float(s)


def extract_run_candidates(raw: bytes) -> list[tuple[bytes, bool]]:
    """Prefix frames plus interior printable runs (payload need not start at bit 0)."""
    out = split_frames(raw)
    seen = {payload for payload, _ in out}
    ascii_map = "".join(chr(b) if 32 <= b < 127 else " " for b in raw)
    for tok in _TOKEN_FIND.findall(ascii_map):
        b = tok.encode("ascii")
        if b not in seen:
            seen.add(b)
            out.append((b, True))
    i = 0
    n = len(raw)
    while i < n:
        if not (0x20 <= raw[i] <= 0x7E):
            i += 1
            continue
        j = i
        while j < n and (0x20 <= raw[j] <= 0x7E or raw[j] in (0x09, 0x0A, 0x0D)):
            j += 1
        chunk = raw[i:j]
        if len(chunk) >= MIN_UNFRAMED_CHARS and chunk not in seen:
            seen.add(chunk)
            out.append((chunk, False))
        i = j + 1
    return out


def split_frames(raw: bytes) -> list[tuple[bytes, bool]]:
    """Return (payload, framed) candidates from packed LSB bytes."""
    out: list[tuple[bytes, bool]] = []
    if not raw:
        return out
    if b"\x00" in raw:
        segs = raw.split(b"\x00")
        for seg in segs[:-1]:
            if seg:
                out.append((seg, True))
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
