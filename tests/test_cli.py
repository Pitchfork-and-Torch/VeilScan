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
    assert "2.3.0" in r.stdout


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


def test_scan_invalid_policy_exits_clean(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(2), style="photo")
    p = tmp_path / "pol.png"
    Image.fromarray(img).save(p)
    r = runner.invoke(app, ["scan", str(p), "--policy", "camra"], color=False)
    assert r.exit_code == 2, r.output
    assert "Traceback" not in r.output
    assert "--policy" in r.output
    assert "generator" in r.output.lower() or "camera" in r.output.lower()


def test_scan_policy_camera_casefold(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(3), style="photo")
    p = tmp_path / "cam.png"
    Image.fromarray(img).save(p)
    r = runner.invoke(app, ["scan", str(p), "--policy", "CAMERA", "--json"])
    assert r.exit_code == 0, r.output
    assert '"policy": "camera"' in r.output


def test_scan_invalid_tier_exits_clean(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(4), style="photo")
    path = tmp_path / "tier.png"
    Image.fromarray(img).save(path)
    r = runner.invoke(app, ["scan", str(path), "--tier", "fastt"], color=False)
    assert r.exit_code == 2, r.output
    assert "Traceback" not in r.output
    assert "--tier" in r.output
    assert "fast" in r.output.lower()


def test_scan_tier_residual_casefold(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(5), style="photo")
    path = tmp_path / "res.png"
    Image.fromarray(img).save(path)
    r = runner.invoke(app, ["scan", str(path), "--tier", "RESIDUAL", "--json"])
    assert r.exit_code == 0, r.output


def test_embed_text_invalid_family_exits_clean(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(6), style="photo")
    src = tmp_path / "src.png"
    Image.fromarray(img).save(src)
    out = tmp_path / "out.png"
    r = runner.invoke(
        app,
        ["embed-text", str(src), str(out), "--message", "hi", "--family", "nope"],
        color=False,
    )
    assert r.exit_code == 2, r.output
    assert "Traceback" not in r.output
    assert "--family" in r.output
    assert "lsb" in r.output.lower()


def test_embed_text_invalid_layout_exits_clean(tmp_path) -> None:
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(7), style="photo")
    src = tmp_path / "src.png"
    Image.fromarray(img).save(src)
    out = tmp_path / "out.png"
    r = runner.invoke(
        app,
        ["embed-text", str(src), str(out), "--message", "hi", "--layout", "nope"],
        color=False,
    )
    assert r.exit_code == 2, r.output
    assert "Traceback" not in r.output
    assert "--layout" in r.output


def test_embed_invalid_family_exits_clean(tmp_path) -> None:
    """embed --family (synthetic WM) must exit 2, distinct from embed-text families."""
    from PIL import Image
    import numpy as np
    from veilscan.generators import synthetic_cover

    img = synthetic_cover(32, 32, np.random.default_rng(8), style="photo")
    src = tmp_path / "src.png"
    Image.fromarray(img).save(src)
    out = tmp_path / "out.png"
    r = runner.invoke(
        app,
        ["embed", str(src), str(out), "--family", "nope"],
        color=False,
    )
    assert r.exit_code == 2, r.output
    assert "Traceback" not in r.output
    assert "--family" in r.output
    assert "lsb" in r.output.lower()

