"""Keyless plaintext recovery. Not a universal watermark decoder. Not a stripper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from veilscan.decode.container import (
    LOSSY_NOTES,
    container_candidates,
    load_raw_rgb,
    lsb_safe_kind,
    sniff_kind,
)
from veilscan.decode.layouts import ACCEPT_SCORE, LSB_LAYOUTS
from veilscan.decode.lsb import extract_bits, pack_bits
from veilscan.decode.score import printable_ratio, score_text, split_frames, utf8_text

Family = Literal["container", "lsb", "none"]
DECODE_SCHEMA = 1


@dataclass
class DecodeCandidate:
    layout: str
    text: str
    score: float
    framed: bool
    family: str

    def to_json(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "text": _clip(self.text, 512),
            "score": round(float(self.score), 6),
            "framed": self.framed,
            "family": self.family,
        }


@dataclass
class DecodeResult:
    found: bool
    family: Family
    text: str | None
    layout: str | None
    confidence: float
    candidates: list[DecodeCandidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    kind: str = "unknown"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": DECODE_SCHEMA,
            "found": self.found,
            "family": self.family,
            "text": self.text,
            "layout": self.layout,
            "confidence": round(float(self.confidence), 6),
            "kind": self.kind,
            "notes": self.notes,
            "candidates": [c.to_json() for c in self.candidates[:3]],
        }


def decode_path(path: str | Path) -> DecodeResult:
    data = Path(path).read_bytes()
    return decode_bytes(data)


def decode_bytes(data: bytes) -> DecodeResult:
    notes: list[str] = []
    kind = sniff_kind(data)
    if kind == "unknown":
        notes.append("Unrecognized image container.")
    if kind in {"jpeg", "webp"}:
        notes.append(LOSSY_NOTES)

    scored: list[DecodeCandidate] = []

    for layout_id, _key, text in container_candidates(data):
        s = score_text(text, framed=True)
        if s <= 0:
            if len(text) >= 2 and printable_ratio(text) >= 0.92:
                s = min(1.0, 0.86 + min(len(text), 20) / 200.0)
            else:
                continue
        scored.append(
            DecodeCandidate(
                layout=layout_id,
                text=text,
                score=s,
                framed=True,
                family="container",
            )
        )

    rgb, _kind, load_notes = load_raw_rgb(data)
    notes.extend(load_notes)
    if rgb is not None and lsb_safe_kind(kind):
        for layout in LSB_LAYOUTS:
            bits = extract_bits(rgb, layout)
            raw = pack_bits(bits, msb_first=bool(layout["msb_first"]))
            for payload, framed in split_frames(raw):
                text = utf8_text(payload)
                if text is None:
                    continue
                s = score_text(text, framed=framed)
                if s < ACCEPT_SCORE:
                    continue
                scored.append(
                    DecodeCandidate(
                        layout=layout["id"],
                        text=text,
                        score=s,
                        framed=framed,
                        family="lsb",
                    )
                )
    elif rgb is not None and kind in {"jpeg", "webp"}:
        notes.append("Skipped spatial LSB on a lossy file.")

    containers = [c for c in scored if c.family == "container" and c.score >= ACCEPT_SCORE]
    lsbs = [c for c in scored if c.family == "lsb" and c.score >= ACCEPT_SCORE]
    if containers:
        containers.sort(key=lambda c: (-c.score, c.layout))
        top = containers[0]
        ordered = containers + lsbs
    else:
        # Table order: first passing LSB layout wins the label.
        top = lsbs[0] if lsbs else None
        ordered = lsbs
    scored = ordered
    found = bool(top and top.score >= ACCEPT_SCORE)
    if not found:
        notes.append(
            "No keyless plaintext. Encrypted stego, vendor IDs (SynthID, Digimarc), "
            "and neural marks will not print here."
        )
        return DecodeResult(
            found=False,
            family="none",
            text=None,
            layout=None,
            confidence=0.0,
            candidates=scored[:3],
            notes=_uniq(notes),
            kind=kind,
        )

    family: Family = "container" if top.family == "container" else "lsb"
    return DecodeResult(
        found=True,
        family=family,
        text=top.text,
        layout=top.layout,
        confidence=float(top.score),
        candidates=scored[:3],
        notes=_uniq(notes),
        kind=kind,
    )


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: n - 1] + "..."


def _uniq(rows: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        out.append(row)
    return out
