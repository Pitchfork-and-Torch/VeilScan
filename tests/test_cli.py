from typer.testing import CliRunner

from veilscan.cli import app

runner = CliRunner()


def test_list_detectors_cli() -> None:
    r = runner.invoke(app, ["list-detectors"])
    assert r.exit_code == 0
    assert "chi_square" in r.stdout
    assert "fsnet_lite" in r.stdout


def test_version_cli() -> None:
    r = runner.invoke(app, ["version"])
    assert r.exit_code == 0
    assert "2.3.1" in r.stdout


def test_bench_write_op_refuses_other_covers(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(1), style="photo")
    Image.fromarray(img).save(tmp_path / "a.png")
    r = runner.invoke(
        app,
        [
            "bench",
            "--covers",
            str(tmp_path),
            "--n",
            "1",
            "--attacks",
            "identity",
            "--write-operating-point",
        ],
    )
    assert r.exit_code == 2, r.stdout + r.stderr
    assert "refusing" in (r.stdout + r.stderr).lower()


def test_robustness_invalid_family_exits_clean() -> None:
    r = runner.invoke(app, ["robustness", "--family", "nope", "--n", "1"])
    assert r.exit_code == 2, r.stdout + r.stderr
    blob = (r.stdout + r.stderr).lower()
    assert "unknown" in blob
    assert "traceback" not in blob


def test_eval_nonpositive_n_exits_clean() -> None:
    """selftest/loao/robustness/calibrate must reject n < 1 with exit 2 (no NaN AUC)."""
    for cmd in (
        ["selftest", "--n", "0"],
        ["loao", "--n", "-1"],
        ["robustness", "--family", "lsb", "--n", "0"],
        ["calibrate", "--n", "-2", "--out", "/tmp/veilscan-cal-reject.json"],
    ):
        r = runner.invoke(app, cmd)
        assert r.exit_code == 2, (cmd, r.exit_code, r.stdout, r.stderr)
        blob = (r.stdout or "") + (r.stderr or "")
        assert "Traceback" not in blob, blob
        assert "--n" in blob or "n" in blob.lower()

