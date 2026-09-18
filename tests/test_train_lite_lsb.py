import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_lite.py"


def test_fsnet_lsb_families_exits_2() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--only", "fsnet_lite", "--families", "lsb,dct", "--steps", "1"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, r.stdout + r.stderr
    assert "must not train on lsb" in (r.stdout + r.stderr)


def test_fsnet_default_families_exits_2() -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--only", "fsnet_lite", "--steps", "1"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 2, r.stdout + r.stderr
