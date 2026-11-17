"""Fetch the frozen public stills pack. Extracted JPEGs stay gitignored."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "camera_covers.manifest.json"
DEFAULT_OUT = ROOT / "data" / "covers" / "camera"
DEFAULT_CACHE = ROOT / "data" / "covers" / "cache"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("kind") not in {"archive", "files"}:
        raise ValueError("manifest must be an object with kind archive|files")
    return data


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(
        url,
        headers={"User-Agent": "VeilScan-cover-fetch/1.0 (research stills; +https://github.com/Pitchfork-and-Torch/VeilScan)"},
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(req, timeout=600) as src, tmp.open("wb") as out:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def _extract_archive(archive: Path, dest: Path, member_glob: str, take: int) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    opener: tarfile.TarFile | zipfile.ZipFile
    if archive.suffix in {".tgz", ".gz"} or archive.name.endswith(".tar.gz"):
        opener = tarfile.open(archive, "r:*")
        members = [m.name.replace("\\", "/") for m in opener.getmembers() if m.isfile()]
        names = _match(members, member_glob)[:take]
        for name in names:
            member = opener.getmember(name)
            target = dest / Path(name).name
            src = opener.extractfile(member)
            if src is None:
                continue
            target.write_bytes(src.read())
        opener.close()
    elif archive.suffix.lower() == ".zip":
        opener = zipfile.ZipFile(archive)
        members = [n.replace("\\", "/") for n in opener.namelist() if not n.endswith("/")]
        names = _match(members, member_glob)[:take]
        for name in names:
            target = dest / Path(name).name
            target.write_bytes(opener.read(name))
        opener.close()
    else:
        raise ValueError(f"unsupported archive {archive}")
    return [dest / Path(n).name for n in names]


def _norm_member(name: str) -> str:
    n = name.replace("\\", "/")
    while n.startswith("./"):
        n = n[2:]
    return n


def _match(names: list[str], pattern: str) -> list[str]:
    from fnmatch import fnmatch

    pat = _norm_member(pattern)
    hit = [n for n in sorted(names) if fnmatch(_norm_member(n), pat)]
    return hit


def fetch(
    manifest_path: Path,
    out_dir: Path,
    cache_dir: Path,
    *,
    dry_run: bool = False,
    allow_empty_hash: bool = False,
) -> dict:
    man = load_manifest(manifest_path)
    url = str(man.get("url") or "")
    if not url:
        raise ValueError("manifest.url missing")
    sha = str(man.get("sha256") or "").strip().lower()
    archive_name = str(man.get("archive") or Path(url).name)
    dest = cache_dir / archive_name
    report = {
        "id": man.get("id"),
        "url": url,
        "archive": str(dest),
        "out": str(out_dir),
        "dry_run": dry_run,
        "files": 0,
    }
    if dry_run:
        report["note"] = "dry-run: no download"
        return report
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        _download(url, dest)
    digest = sha256_file(dest)
    report["sha256"] = digest
    if sha:
        if digest != sha:
            dest.unlink(missing_ok=True)
            raise ValueError(f"sha256 mismatch: got {digest} expected {sha}")
    elif not allow_empty_hash:
        raise ValueError("manifest.sha256 empty; pass --allow-empty-hash to record the first digest")
    take = int(man.get("take") or 64)
    glob = str(man.get("member_glob") or "*.jpg")
    files = _extract_archive(dest, out_dir, glob, take)
    report["files"] = len(files)
    if len(files) < min(take, 8):
        raise ValueError(f"extracted {len(files)} files, expected up to {take}")
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fetch VeilScan camera stills pack (gitignored).")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-empty-hash", action="store_true")
    args = p.parse_args(argv)
    try:
        report = fetch(
            args.manifest,
            args.out,
            args.cache,
            dry_run=args.dry_run,
            allow_empty_hash=args.allow_empty_hash,
        )
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
