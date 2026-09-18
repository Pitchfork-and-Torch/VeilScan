"""Flag harvest. Hunt uses this, not decode looks_like_token."""

from __future__ import annotations

import re

DEFAULT_FLAG_RE = (
    r"(?:FLAG|CTF|PICOCTF|HTB|THM|picoCTF|flag|ctf)"
    r"\{[^\r\n]{1,256}?\}"
)

_DEFAULT = re.compile(DEFAULT_FLAG_RE, re.IGNORECASE)


def compile_flag_re(pattern: str | None) -> re.Pattern[str]:
    if not pattern:
        return _DEFAULT
    return re.compile(pattern, re.IGNORECASE)


def find_flags(text: str, cre: re.Pattern[str] | None = None) -> list[str]:
    if not text:
        return []
    rx = cre or _DEFAULT
    out: list[str] = []
    seen: set[str] = set()
    for m in rx.finditer(text):
        g = m.group(0)
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out


def find_flags_bytes(data: bytes, cre: re.Pattern[str] | None = None) -> list[str]:
    if not data:
        return []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", "replace")
    return find_flags(text, cre)
