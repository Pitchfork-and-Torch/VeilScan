"""Keyless plaintext recovery. Not a universal watermark decoder. Not a stripper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from veilscan.decode.bitplane_qr import decode_lsb_qr
from veilscan.decode.container import (
    LOSSY_NOTES,
    container_candidates,
    load_raw_rgb,
    lsb_safe_kind,
    sniff_kind,
)
from veilscan.decode.hotspots import Hotspot, find_lsb_hotspots, iter_patch_windows
from veilscan.decode.jsteg import iter_jsteg_payloads
from veilscan.decode.layouts import ACCEPT_SCORE, LSB_LAYOUTS
from veilscan.decode.lsb import extract_bits, pack_bits
from veilscan.decode.score import (
    extract_run_candidates,
    printable_ratio,
    score_text,
    utf8_text,
)

Family = Literal["container", "lsb", "qr", "jsteg", "none"]
DECODE_SCHEMA = 1


@dataclass
class DecodeCandidate:
    layout: str
    text: str
    score: float
    framed: bool
    family: str
    bbox: tuple[int, int, int, int] | None = None

    def to_json(self) -> dict[str, Any]:
        d = {
            "layout": self.layout,
            "text": _clip(self.text, 512),
            "score": round(float(self.score), 6),
            "framed": self.framed,
            "family": self.family,
        }
        if self.bbox:
            d["bbox"] = list(self.bbox)
        return d


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
    hotspots: list[Hotspot] = field(default_factory=list)

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
            "hotspots": [h.to_json() for h in self.hotspots[:8]],
            "candidates": [c.to_json() for c in self.candidates[:5]],
        }


def decode_path(path: str | Path) -> DecodeResult:
    data = Path(path).read_bytes()
    return decode_bytes(data, source=str(path))


def decode_bytes(data: bytes, source: str | None = None) -> DecodeResult:
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
    hotspots: list[Hotspot] = []

    if rgb is not None:
        if lsb_safe_kind(kind):
            _collect_lsb(rgb, scored, bbox=None, offsets=(0,))
        else:
            notes.append("Decoded pixels used for localized LSB hunt. JPEG/WebP may have damaged exact bits.")

        hotspots = find_lsb_hotspots(rgb)
        seen_box: set[tuple[int, int, int, int]] = set()
        qr_done: set[tuple[int, int, int, int]] = set()
        dense = lsb_safe_kind(kind)
        for hs in hotspots:
            if _strong_payload(scored):
                break
            windows = iter_patch_windows(rgb, hs) if dense else ()
            for win in windows:
                box = (win.x, win.y, win.w, win.h)
                if box in seen_box:
                    continue
                seen_box.add(box)
                crop = win.crop(rgb)
                if crop.size == 0 or min(crop.shape[:2]) < 8:
                    continue
                _collect_lsb(
                    crop,
                    scored,
                    bbox=box,
                    offsets=(0,),
                    layouts=LSB_LAYOUTS[:6],
                )
                if _strong_payload(scored):
                    break
            pad = hs.pad(rgb, margin=16)
            pbox = (pad.x, pad.y, pad.w, pad.h)
            if pbox not in qr_done and not _strong_payload(scored):
                qr_done.add(pbox)
                for qtext in decode_lsb_qr(pad.crop(rgb)):
                    s = max(score_text(qtext, framed=True), 0.92)
                    scored.append(
                        DecodeCandidate(
                            layout="qr-bitplane",
                            text=qtext,
                            score=s,
                            framed=True,
                            family="qr",
                            bbox=pbox,
                        )
                    )

        jsteg_src: str | bytes = source if source else data
        if kind == "jpeg":
            boxes: list[tuple[int, int, int, int] | None] = [None]
            boxes.extend((h.x, h.y, h.w, h.h) for h in hotspots[:4])
            for box in boxes:
                for layout_id, raw in iter_jsteg_payloads(jsteg_src, box=box):
                    _collect_from_raw(raw, layout_id, "jsteg", scored, bbox=box)

    containers = [c for c in scored if c.family == "container" and c.score >= ACCEPT_SCORE]
    others = [c for c in scored if c.family != "container" and c.score >= ACCEPT_SCORE]
    others.sort(key=lambda c: (-c.score, -len(c.text), c.layout))
    if containers:
        containers.sort(key=lambda c: (-c.score, -len(c.text), c.layout))
        top = containers[0]
        ordered = containers + others
    else:
        top = others[0] if others else None
        ordered = others

    found = bool(top and top.score >= ACCEPT_SCORE)
    if hotspots:
        top_hs = hotspots[0]
        notes.append(
            f"LSB hotspot at ({top_hs.x},{top_hs.y}) {top_hs.w}x{top_hs.h} "
            f"corr={top_hs.corr:.3f} (blind tile hunt, no prior box)."
        )
    if not found:
        notes.append(
            "No keyless plaintext. Encrypted stego, vendor IDs (SynthID, Digimarc), "
            "and neural marks will not print here."
        )
        if kind in {"jpeg", "webp"} and hotspots:
            notes.append(
                "A localized LSB anomaly was found. If the original was a PNG, "
                "the payload may still be intact there. This JPEG/WebP likely damaged the bits."
            )
        return DecodeResult(
            found=False,
            family="none",
            text=None,
            layout=None,
            confidence=0.0,
            candidates=ordered[:5],
            notes=_uniq(notes),
            kind=kind,
            hotspots=hotspots,
        )

    fam: Family
    if top.family in ("container", "lsb", "qr", "jsteg"):
        fam = top.family  # type: ignore[assignment]
    else:
        fam = "lsb"
    return DecodeResult(
        found=True,
        family=fam,
        text=top.text,
        layout=top.layout,
        confidence=float(top.score),
        candidates=ordered[:5],
        notes=_uniq(notes),
        kind=kind,
        hotspots=hotspots,
    )


def _strong_payload(scored: list[DecodeCandidate]) -> bool:
    return any(c.score >= 0.90 and len(c.text) >= 8 and c.family in {"lsb", "qr", "jsteg"} for c in scored)


def _collect_lsb(
    rgb,
    scored: list[DecodeCandidate],
    bbox: tuple[int, int, int, int] | None,
    offsets: tuple[int, ...],
    layouts=None,
) -> None:
    for layout in layouts or LSB_LAYOUTS:
        bits = extract_bits(rgb, layout)
        for off in offsets:
            chunk = bits[off:] if off else bits
            raw = pack_bits(chunk, msb_first=bool(layout["msb_first"]))
            tag = layout["id"] if off == 0 else f"{layout['id']}+{off}"
            _collect_from_raw(raw, tag, "lsb", scored, bbox=bbox)


def _collect_from_raw(
    raw: bytes,
    layout_id: str,
    family: str,
    scored: list[DecodeCandidate],
    bbox: tuple[int, int, int, int] | None,
) -> None:
    for payload, framed in extract_run_candidates(raw):
        text = utf8_text(payload)
        if text is None:
            continue
        s = score_text(text, framed=framed)
        if s < ACCEPT_SCORE:
            continue
        scored.append(
            DecodeCandidate(
                layout=layout_id,
                text=text,
                score=s,
                framed=framed,
                family=family,
                bbox=bbox,
            )
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
