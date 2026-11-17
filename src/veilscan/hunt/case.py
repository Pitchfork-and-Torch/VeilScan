"""Folder hunt. One case.json an agent can parse. Not a cloud job."""

from __future__ import annotations

import json
from pathlib import Path

CASE_SCHEMA = 1
CASE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".wav",
    ".pdf",
    ".mp3",
    ".flac",
}
_MAX_BYTES = 8 * 1024 * 1024
_MAX_FILES = 200


def _safe(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in name)
    return cleaned[:80] or "file"


def hunt_case(
    folder: str | Path,
    *,
    out_dir: Path | None = None,
    flag_re: str | None = None,
    deep: bool = False,
    wordlist: Path | None = None,
) -> dict:
    from veilscan.hunt.pipeline import hunt_path

    root = Path(folder)
    dest = Path(out_dir) if out_dir else None
    if dest is not None:
        dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    flags: list[str] = []
    seen: set[str] = set()
    notes: list[str] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if dest is not None and dest in path.parents:
            continue
        if path.suffix.lower() not in CASE_EXTS:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > _MAX_BYTES or size <= 0:
            notes.append(f"skip {path.name} size={size}")
            continue
        if scanned >= _MAX_FILES:
            notes.append(f"stopped at {_MAX_FILES} files")
            break
        scanned += 1
        file_out = None
        if dest is not None:
            file_out = dest / _safe(path.stem)
        result = hunt_path(
            path,
            out_dir=file_out,
            flag_re=flag_re,
            deep=deep,
            wordlist=wordlist,
        )
        for flag in result.flags:
            if flag not in seen:
                seen.add(flag)
                flags.append(flag)
        rows.append(
            {
                "path": str(path),
                "name": path.name,
                "kind": result.kind,
                "size": result.size,
                "status": result.status(),
                "summary": result.summary(),
                "flags": list(result.flags),
                "elapsed_ms": round(float(result.elapsed_ms), 1),
            }
        )
    status = "flags" if flags else ("findings" if any(r["status"] != "empty" for r in rows) else "empty")
    report = {
        "schema_version": CASE_SCHEMA,
        "path": str(root),
        "status": status,
        "file_count": len(rows),
        "flag_count": len(flags),
        "flags": flags,
        "summary": _summary(flags, rows),
        "notes": notes,
        "files": rows,
    }
    if dest is not None:
        (dest / "case.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lines = [
            f"VeilScan case  files={len(rows)}  flags={len(flags)}",
            f"flags: {', '.join(flags) if flags else '(none)'}",
            "",
        ]
        for row in rows:
            lines.append(f"[{row['status']}] {row['name']}  {row['summary']}")
        (dest / "case.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def _summary(flags: list[str], rows: list[dict]) -> str:
    if flags:
        if len(flags) == 1:
            return f"1 flag  {flags[0]}"
        return f"{len(flags)} flags  {flags[0]}"
    if rows:
        return f"{len(rows)} files  no FLAG{{}}"
    return "no files"
