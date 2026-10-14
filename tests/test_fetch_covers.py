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


def test_fetch_zip_png_glob(tmp_path: Path) -> None:
    import zipfile

    mod = _load_fetch()
    inner = "DIV2K_valid_HR"
    src = tmp_path / "src" / inner
    src.mkdir(parents=True)
    for i in range(5):
        img = synthetic_cover(32, 32, np.random.default_rng(i), style="photo")
        Image.fromarray(img).save(src / f"{801 + i:04d}.png")
    archive = tmp_path / "DIV2K_valid_HR.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in src.glob("*.png"):
            zf.write(p, arcname=f"{inner}/{p.name}")
    digest = mod.sha256_file(archive)
    man = {
        "kind": "archive",
        "url": archive.as_uri(),
        "archive": archive.name,
        "sha256": digest,
        "member_glob": "DIV2K_valid_HR/*.png",
        "take": 5,
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    out = tmp_path / "out"
    report = mod.fetch(man_path, out, cache)
    names = sorted(p.name for p in out.glob("*.png"))
    assert report["files"] == 5
    assert names == [f"{801 + i:04d}.png" for i in range(5)]


def test_fetch_train_glob_not_test(tmp_path: Path) -> None:
    mod = _load_fetch()
    inner_train = "BSR/BSDS500/data/images/train"
    inner_test = "BSR/BSDS500/data/images/test"
    src_dir = tmp_path / "src"
    (src_dir / inner_train).mkdir(parents=True)
    (src_dir / inner_test).mkdir(parents=True)
    for i in range(6):
        img = synthetic_cover(48, 48, np.random.default_rng(i), style="photo")
        Image.fromarray(img).save(src_dir / inner_train / f"tr{i:04d}.jpg", quality=90)
    for i in range(4):
        img = synthetic_cover(48, 48, np.random.default_rng(80 + i), style="photo")
        Image.fromarray(img).save(src_dir / inner_test / f"te{i:04d}.jpg", quality=90)
    archive = tmp_path / "pack.tgz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src_dir / "BSR", arcname="BSR")
    digest = mod.sha256_file(archive)
    man = {
        "kind": "archive",
        "url": archive.as_uri(),
        "archive": archive.name,
        "sha256": digest,
        "member_glob": "BSR/BSDS500/data/images/train/*.jpg",
        "take": 6,
    }
    man_path = tmp_path / "man.json"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / archive.name).write_bytes(archive.read_bytes())
    out = tmp_path / "out"
    report = mod.fetch(man_path, out, cache)
    names = sorted(p.name for p in out.glob("*.jpg"))
    assert report["files"] == 6
    assert all(n.startswith("tr") for n in names)
    assert not any(n.startswith("te") for n in names)


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
