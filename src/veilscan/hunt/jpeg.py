"""Native JPEG JSteg extract. Optional jpeglib. Not F5/OutGuess."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from veilscan.decode.jsteg import available, iter_jsteg_payloads
from veilscan.hunt.payloads import inspect_payload
from veilscan.hunt.types import HuntFinding


def jsteg_available() -> bool:
    return available()


def iter_jsteg_findings(
    data: bytes,
    source: str | None,
    cre,
    *,
    stop_on_flag: bool = False,
) -> Iterator[HuntFinding]:
    if not available():
        return
    src: str | bytes = source if source and Path(source).is_file() else data
    seen: set[str] = set()
    n_out = 0
    for layout, raw in iter_jsteg_payloads(src):
        if not raw:
            continue
        for hit in inspect_payload(raw, cre):
            text = str(hit.get("text") or "")
            flags = list(hit.get("flags") or [])
            key = layout + text[:160]
            if key in seen:
                continue
            seen.add(key)
            n_out += 1
            yield HuntFinding(
                family="jsteg",
                method=layout,
                confidence=float(hit.get("confidence") or 0.85),
                evidence=str(hit.get("kind") or "dct-ac-lsb"),
                text=text or None,
                flag_hit=bool(flags),
                extra={"flags": flags, "kind": hit.get("kind")},
            )
            if stop_on_flag and flags:
                return
            if n_out >= 16:
                return
