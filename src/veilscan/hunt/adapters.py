"""Optional PATH adapters for known JPEG stego tools. Never required for gym."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from veilscan.hunt.flags import find_flags_bytes
from veilscan.hunt.types import HuntFinding

_TIMEOUT = 20
_MAX_PASSWORDS = 64
_TOOLS = ("stegseek", "steghide", "outguess", "jpseek")


def which_tools() -> dict[str, str | None]:
    return {name: shutil.which(name) for name in _TOOLS}


def _passwords(wordlist: Path | None) -> list[str]:
    out = [""]
    if wordlist is None or not Path(wordlist).is_file():
        return out
    text = Path(wordlist).read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        pw = line.strip()
        if pw and pw not in out:
            out.append(pw)
        if len(out) >= _MAX_PASSWORDS:
            break
    return out


def _run(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            argv,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def _finding(method: str, text: str, flags: list[str], evidence: str) -> HuntFinding:
    return HuntFinding(
        family="adapter",
        method=method,
        confidence=0.98 if flags else 0.7,
        evidence=evidence,
        text=text[:4096] if text else None,
        flag_hit=bool(flags),
        extra={"flags": flags},
    )


def run_adapters(
    image: Path,
    *,
    wordlist: Path | None = None,
    dest: Path | None = None,
    cre=None,
) -> tuple[list[HuntFinding], list[str]]:
    notes: list[str] = []
    findings: list[HuntFinding] = []
    tools = which_tools()
    present = {k: v for k, v in tools.items() if v}
    if not present:
        notes.append("No PATH stego adapters (stegseek, steghide, outguess, jpseek).")
        if wordlist is not None:
            notes.append("wordlist unused (no adapter binary)")
        return findings, notes

    work = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="veilscan-adapt-"))
    work.mkdir(parents=True, exist_ok=True)
    passwords = _passwords(wordlist)

    if tools.get("stegseek"):
        if wordlist is not None and Path(wordlist).is_file():
            outp = work / "stegseek.out"
            proc = _run([tools["stegseek"], str(image), str(wordlist), "-xf", str(outp), "-f"])
            blob = outp.read_bytes() if outp.is_file() else b""
            if not blob and proc and proc.stdout:
                blob = proc.stdout.encode("utf-8", "replace")
            if blob:
                flags = find_flags_bytes(blob, cre)
                text = blob.decode("utf-8", "replace")
                findings.append(_finding("stegseek", text, flags, "stegseek wordlist extract"))
        seed_out = work / "stegseek-seed.out"
        proc = _run([tools["stegseek"], "--seed", str(image), "-xf", str(seed_out), "-f"])
        blob = seed_out.read_bytes() if seed_out.is_file() else b""
        if blob:
            flags = find_flags_bytes(blob, cre)
            text = blob.decode("utf-8", "replace")
            findings.append(_finding("stegseek-seed", text, flags, "stegseek --seed (unencrypted steghide)"))
        elif proc is None:
            notes.append("stegseek --seed timed out or failed")

    if tools.get("steghide"):
        for pw in passwords:
            outp = work / "steghide.out"
            if outp.exists():
                outp.unlink()
            argv = [tools["steghide"], "extract", "-sf", str(image), "-xf", str(outp), "-f", "-p", pw]
            _run(argv)
            if outp.is_file() and outp.stat().st_size:
                blob = outp.read_bytes()
                flags = find_flags_bytes(blob, cre)
                text = blob.decode("utf-8", "replace")
                findings.append(_finding("steghide", text, flags, f"steghide extract p={pw!r}" if pw else "steghide empty password"))
                break
        else:
            notes.append("steghide present; no extract with empty password or wordlist")

    if tools.get("outguess"):
        outp = work / "outguess.out"
        tried = False
        for pw in passwords[:8]:
            if outp.exists():
                outp.unlink()
            argv = [tools["outguess"], "-r", str(image), str(outp)] if not pw else [tools["outguess"], "-k", pw, "-r", str(image), str(outp)]
            _run(argv)
            tried = True
            if outp.is_file() and outp.stat().st_size:
                blob = outp.read_bytes()
                flags = find_flags_bytes(blob, cre)
                text = blob.decode("utf-8", "replace")
                findings.append(_finding("outguess", text, flags, "outguess -r"))
                break
        if tried and not any(f.method == "outguess" for f in findings):
            notes.append("outguess present; no extract")

    if tools.get("jpseek"):
        outp = work / "jpseek.out"
        _run([tools["jpseek"], str(image), str(outp)])
        if outp.is_file() and outp.stat().st_size:
            blob = outp.read_bytes()
            flags = find_flags_bytes(blob, cre)
            text = blob.decode("utf-8", "replace")
            findings.append(_finding("jpseek", text, flags, "jpseek extract"))

    if not findings:
        notes.append("PATH adapters ran; no payload extracted")
    return findings, notes


def plant_steghide(cover_jpeg: Path, payload: bytes, password: str, dest: Path) -> bool:
    exe = shutil.which("steghide")
    if not exe:
        return False
    payload_path = dest.with_suffix(".payload.txt")
    payload_path.write_bytes(payload)
    proc = _run(
        [
            exe,
            "embed",
            "-cf",
            str(cover_jpeg),
            "-ef",
            str(payload_path),
            "-sf",
            str(dest),
            "-p",
            password,
            "-f",
        ]
    )
    return dest.is_file() and dest.stat().st_size > 0 and proc is not None and proc.returncode == 0
