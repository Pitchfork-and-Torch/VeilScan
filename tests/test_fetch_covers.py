import json
import tarfile
from pathlib import Path

from PIL import Image
import numpy as np

from veilscan.generators import synthetic_cover


def _load_fetch():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_camera_covers.py"
    spec = importlib.util.spec_from_file_location("fetch_camera_covers", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_fetch_local_archive(tmp_path: Path) -> None:
    mod = _load_fetch()
    inner = "BSR/BSDS500/data/images/test"
    src_dir = tmp_path / "src"
    (src_dir / inner).mkdir(parents=True)
    for i in range(8):
        img = synthetic_cover(64, 64, np.random.default_rng(i), style="photo")
        Image.fromarray(img).save(src_dir / inner / f"{i:04d}.jpg", quality=90)
    archive = tmp_path / "pack.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src_dir / "BSR", arcname="BSR")
    digest = mod.sha256_file(archive)
    man = {
        "schema_version": 1,
        "id": "fixture",
        "kind": "archive",
        "url": archive.as_uri(),
        "archive": archive.name,
        "sha256": digest,
        "member_glob": "BSR/BSDS500/data/images/test/*.jpg",
        "take": 8,
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    out = tmp_path / "out"
    cache = tmp_path / "cache"
    cache.mkdir()
    # local file url may not download; copy into cache under expected name
    cached = cache / archive.name
    cached.write_bytes(archive.read_bytes())
    report = mod.fetch(man_path, out, cache, dry_run=False, allow_empty_hash=False)
    assert report["files"] == 8
    assert len(list(out.glob("*.jpg"))) == 8


def test_fetch_hash_mismatch(tmp_path: Path) -> None:
    mod = _load_fetch()
    archive = tmp_path / "pack.tgz"
    archive.write_bytes(b"not-a-tar")
    man = {
        "kind": "archive",
        "url": "http://example.invalid/pack.tgz",
        "archive": "pack.tgz",
        "sha256": "0" * 64,
        "member_glob": "*.jpg",
        "take": 1,
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pack.tgz").write_bytes(archive.read_bytes())
    try:
        mod.fetch(man_path, tmp_path / "out", cache)
        raise AssertionError("expected hash mismatch")
    except ValueError as exc:
        assert "sha256" in str(exc).lower()


def test_empty_hash_requires_flag(tmp_path: Path) -> None:
    mod = _load_fetch()
    archive = tmp_path / "pack.tgz"
    archive.write_bytes(b"placeholder")
    man = {
        "kind": "archive",
        "url": "http://example.invalid/pack.tgz",
        "archive": "pack.tgz",
        "sha256": "",
        "member_glob": "*.jpg",
        "take": 1,
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "pack.tgz").write_bytes(archive.read_bytes())
    try:
        mod.fetch(man_path, tmp_path / "out", cache, allow_empty_hash=False)
        raise AssertionError("expected empty-hash refusal")
    except ValueError as exc:
        assert "empty" in str(exc).lower()


def test_dry_run_no_network(tmp_path: Path) -> None:
    mod = _load_fetch()
    man = {
        "kind": "archive",
        "id": "x",
        "url": "https://example.invalid/nope.tgz",
        "archive": "nope.tgz",
        "sha256": "",
        "member_glob": "*.jpg",
        "take": 4,
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    report = mod.fetch(man_path, tmp_path / "out", tmp_path / "cache", dry_run=True)
    assert report["dry_run"] is True
    assert report["files"] == 0
