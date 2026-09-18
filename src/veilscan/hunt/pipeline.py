"""Ordered hunt: identity, chunks, carve, strings, flags. No Linux binaries required."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from veilscan.decode.container import lsb_safe_kind, sniff_kind
from veilscan.hunt.bitplanes import qr_findings, render_bitplane_sheet
from veilscan.hunt.carve import carve
from veilscan.hunt.chunks import extract_chunks
from veilscan.hunt.flags import compile_flag_re, find_flags, find_flags_bytes
from veilscan.hunt.image import load_hunt_image
from veilscan.hunt.palette import iter_palette_findings
from veilscan.hunt.strings import extract_strings
from veilscan.hunt.types import HuntFinding, HuntResult
from veilscan.hunt.zsteg import iter_zsteg_findings


def hunt_path(
    path: str | Path,
    *,
    out_dir: Optional[Path] = None,
    flag_re: Optional[str] = None,
    stop_on_flag: bool = False,
    wordlist: Optional[Path] = None,
    deep: bool = False,
) -> HuntResult:
    p = Path(path)
    data = p.read_bytes()
    result = hunt_bytes(
        data,
        source=str(p),
        out_dir=out_dir,
        flag_re=flag_re,
        stop_on_flag=stop_on_flag,
        wordlist=wordlist,
        deep=deep,
    )
    result.path = str(p)
    return result


def hunt_bytes(
    data: bytes,
    source: str | None = None,
    *,
    out_dir: Optional[Path] = None,
    flag_re: Optional[str] = None,
    stop_on_flag: bool = False,
    wordlist: Optional[Path] = None,
    deep: bool = False,
) -> HuntResult:
    t0 = time.perf_counter()
    cre = compile_flag_re(flag_re)
    kind = sniff_kind(data)
    result = HuntResult(
        path=source,
        sha256=hashlib.sha256(data).hexdigest(),
        kind=kind,
        size=len(data),
    )
    if kind == "jpeg":
        result.notes.append("F5/nsF5 payload extract is not implemented.")

    dest = Path(out_dir) if out_dir else None
    if dest:
        dest.mkdir(parents=True, exist_ok=True)

    flags: list[str] = []
    seen_flag: set[str] = set()

    def add_flags(hits: list[str]) -> None:
        for f in hits:
            if f not in seen_flag:
                seen_flag.add(f)
                flags.append(f)

    def maybe_stop() -> bool:
        return bool(stop_on_flag and flags)

    for ch in extract_chunks(data):
        hits = find_flags(ch.text, cre)
        add_flags(hits)
        result.findings.append(
            HuntFinding(
                family=ch.family,
                method=ch.method,
                confidence=0.95 if hits else 0.80,
                evidence=ch.extra or ch.key,
                text=ch.text,
                offset=ch.offset,
                length=len(ch.text),
                flag_hit=bool(hits),
                extra={"key": ch.key} if ch.key else {},
            )
        )
        if maybe_stop():
            return _finish(result, flags, t0, dest)

    for hit in carve(data):
        blob_flags = find_flags_bytes(hit.payload, cre)
        add_flags(blob_flags)
        artifact = None
        if dest is not None:
            artifact = f"carve_{hit.kind}_{hit.offset}.bin"
            (dest / artifact).write_bytes(hit.payload[: 8 * 1024 * 1024])
            result.artifacts.append(artifact)
        result.findings.append(
            HuntFinding(
                family="carve",
                method=hit.kind,
                confidence=0.97 if hit.inner_files or blob_flags else 0.85,
                evidence=hit.note,
                offset=hit.offset,
                length=hit.length,
                artifact_name=artifact,
                flag_hit=bool(blob_flags),
            )
        )
        for name, raw in hit.inner_files:
            inner_hits = find_flags_bytes(raw, cre)
            add_flags(inner_hits)
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("latin-1", "replace") if len(raw) <= 8192 else None
            inner_art = None
            if dest is not None:
                safe = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name) or "inner"
                inner_art = f"zip_{hit.offset}_{safe}"
                (dest / inner_art).write_bytes(raw[: 8 * 1024 * 1024])
                result.artifacts.append(inner_art)
            result.findings.append(
                HuntFinding(
                    family="carve",
                    method="zip-member",
                    confidence=0.99 if inner_hits else 0.90,
                    evidence=f"{name} inside {hit.kind} @ {hit.offset}",
                    text=text if text and len(text) <= 4096 else None,
                    artifact_name=inner_art,
                    flag_hit=bool(inner_hits),
                    extra={"member": name},
                )
            )
        if maybe_stop():
            return _finish(result, flags, t0, dest)

    for s in extract_strings(data):
        hits = find_flags(s, cre)
        if not hits and len(s) < 12:
            continue
        if hits:
            add_flags(hits)
            result.findings.append(
                HuntFinding(
                    family="strings",
                    method="ascii-run",
                    confidence=0.92,
                    evidence="printable run on raw bytes",
                    text=s,
                    flag_hit=True,
                )
            )
            if maybe_stop():
                return _finish(result, flags, t0, dest)

    raw_flags = find_flags_bytes(data, cre)
    add_flags(raw_flags)
    if maybe_stop():
        return _finish(result, flags, t0, dest)

    def ingest(f: HuntFinding) -> None:
        extra_flags = list((f.extra or {}).get("flags") or [])
        if extra_flags:
            add_flags(extra_flags)
        elif f.text:
            add_flags(find_flags(f.text, cre))
        result.findings.append(f)

    if kind == "jpeg":
        from veilscan.hunt.jpeg import iter_jsteg_findings, jsteg_available

        if not jsteg_available():
            result.notes.append("jpeglib missing; JSteg extract skipped")
        else:
            for f in iter_jsteg_findings(data, source, cre, stop_on_flag=stop_on_flag):
                ingest(f)
                if maybe_stop():
                    return _finish(result, flags, t0, dest)
        from veilscan.hunt.adapters import run_adapters

        img_path = Path(source) if source and Path(source).is_file() else None
        if img_path is None and dest is not None:
            img_path = dest / "input.jpg"
            img_path.write_bytes(data)
        if img_path is not None:
            af, notes = run_adapters(img_path, wordlist=wordlist, dest=dest, cre=cre)
            result.notes.extend(notes)
            for f in af:
                ingest(f)
                if maybe_stop():
                    return _finish(result, flags, t0, dest)
        elif wordlist is not None:
            result.notes.append("wordlist unused (need a filesystem JPEG path for adapters)")

    index, rgba, load_notes = load_hunt_image(data)
    result.notes.extend(load_notes)

    if index is not None:
        for f in iter_palette_findings(index, cre, deep=deep, stop_on_flag=stop_on_flag):
            ingest(f)
            if maybe_stop():
                return _finish(result, flags, t0, dest)

    if rgba is not None and lsb_safe_kind(kind) and index is None:
        for f in iter_zsteg_findings(rgba, cre, deep=deep, stop_on_flag=stop_on_flag):
            ingest(f)
            if maybe_stop():
                return _finish(result, flags, t0, dest)

    if rgba is not None:
        for f in qr_findings(rgba, cre):
            ingest(f)
            if maybe_stop():
                return _finish(result, flags, t0, dest)

    if dest is not None:
        if index is not None:
            sheet = dest / "bitplanes.png"
            render_bitplane_sheet(index, sheet)
            result.artifacts.append("bitplanes.png")
        elif rgba is not None:
            sheet = dest / "bitplanes.png"
            render_bitplane_sheet(rgba, sheet)
            result.artifacts.append("bitplanes.png")

    if not result.findings and not flags:
        result.notes.append("No container text, trailing payload, LSB bitstream, or FLAG{} found.")
        result.notes.append("Encrypted stego and keyed JPEG tools need a later hunt phase or --wordlist adapter.")

    return _finish(result, flags, t0, dest)


def _finish(result: HuntResult, flags: list[str], t0: float, dest: Path | None) -> HuntResult:
    result.flags = flags
    result.elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if dest is not None:
        import json

        lines = [
            f"VeilScan hunt  sha256={result.sha256}",
            f"kind={result.kind} size={result.size} elapsed_ms={result.elapsed_ms:.1f}",
            f"flags: {', '.join(flags) if flags else '(none)'}",
            "",
        ]
        for f in result.findings:
            bit = "FLAG" if f.flag_hit else "hit"
            preview = (f.text or "")[:120].replace("\n", " ")
            lines.append(f"[{bit}] {f.family}/{f.method} {f.evidence} {preview}")
        (dest / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result.artifacts.append("findings.json")
        result.artifacts.append("report.txt")
        rgb = None
        try:
            from veilscan.hunt.image import load_hunt_image

            src_bytes = Path(result.path).read_bytes() if result.path and Path(result.path).is_file() else None
            if src_bytes:
                _idx, rgba, _ = load_hunt_image(src_bytes)
                if rgba is not None:
                    rgb = rgba
        except Exception:
            rgb = None
        from veilscan.hunt.hud import render_hunt_report

        bp = dest / "bitplanes.png"
        render_hunt_report(
            result,
            dest / "report.png",
            rgb=rgb,
            bitplane_path=bp if bp.is_file() else None,
        )
        result.artifacts.append("report.png")
        payload = result.to_json()
        (dest / "findings.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return result
