from pathlib import Path

import numpy as np
from PIL import Image

from veilscan.generators import FAMILIES


def _train_mod():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_lite.py"
    spec = importlib.util.spec_from_file_location("veilscan_train_lite", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_split_cover_paths_disjoint() -> None:
    mod = _train_mod()
    paths = [Path(f"{i}.jpg") for i in range(10)]
    train, val = mod.split_cover_paths(paths, seed=0, val_frac=0.2)
    assert len(val) == 2
    assert len(train) == 8
    assert set(train).isdisjoint(val)
    assert set(train) | set(val) == set(paths)


def test_refuse_frozen_test_pack() -> None:
    mod = _train_mod()
    try:
        mod.assert_not_frozen_test(mod.FROZEN_TEST_COVERS, allow=False)
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        msg = str(exc).lower()
        assert "frozen" in msg or "refuse" in msg
    mod.assert_not_frozen_test(mod.FROZEN_TEST_COVERS, allow=True)


def test_disk_cover_mean(tmp_path: Path) -> None:
    mod = _train_mod()
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[:] = (12, 220, 40)
    for name in ("a.jpg", "b.jpg"):
        Image.fromarray(img).save(tmp_path / name, quality=95)
    paths = sorted(tmp_path.glob("*.jpg"))
    ds = mod.SyntheticWM(
        24,
        seed=0,
        jpeg_prob=0.0,
        size=32,
        families=["dct"],
        cover_paths=paths,
        cover_mix=0.0,
    )
    assert "lsb" not in ds.families
    cover_x = None
    for i in range(len(ds)):
        x, y = ds[i]
        if float(y) < 0.5:
            cover_x = x.numpy()
            break
    assert cover_x is not None
    # disk stills are green-heavy; synthetic photo/sine would not pin G this high
    assert float(cover_x[1].mean()) > 0.7
    assert float(cover_x[0].mean()) < 0.2


def test_fsnet_covers_with_lsb_still_exits_2(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = 80
    Image.fromarray(img).save(tmp_path / "c.jpg", quality=90)
    r = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "train_lite.py"),
            "--only",
            "fsnet_lite",
            "--families",
            "lsb,dct",
            "--steps",
            "1",
            "--covers",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "must not train on lsb" in (r.stdout + r.stderr)


def test_families_without_lsb_keeps_frequency_set() -> None:
    mod = _train_mod()
    ds = mod.SyntheticWM(4, seed=1, families=["dct", "spread", "dwt", "tree_ring"], holdout=None)
    assert set(ds.families) <= set(FAMILIES)
    assert "lsb" not in ds.families
