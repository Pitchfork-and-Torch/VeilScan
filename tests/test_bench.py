from pathlib import Path

from PIL import Image

from veilscan.eval.bench import DEFAULT_OP, load_camera_covers, load_protocol, render_markdown, run_bench, write_outputs
from veilscan.eval.metrics import cut_at_fpr, fpr_at
from veilscan.generators import synthetic_cover
import numpy as np


def test_fpr_at_named_thresholds() -> None:
    from veilscan.eval.bench import fpr_at_named_thresholds

    scores = np.asarray([0.10, 0.20, 0.30, 0.80, 0.90])
    out = fpr_at_named_thresholds(scores, {"low": 0.15, "high": 0.95})
    assert out["low"] == 0.8
    assert out["high"] == 0.0


def test_cut_at_fpr_monotonic() -> None:
    covers = np.linspace(0.0, 0.4, 20)
    t = cut_at_fpr(covers, 0.05)
    fpr = fpr_at(np.zeros(20, dtype=np.int32), covers, t)
    assert fpr <= 0.05 + 1e-9


def test_bench_smoke(tmp_path: Path) -> None:
    proto = load_protocol()
    report = run_bench(
        proto,
        n=2,
        size=64,
        styles=["sine"],
        families=["lsb", "dct"],
        attacks=["identity"],
        detectors=["chi_square", "dct", "rs_analysis"],
    )
    assert report["n"] == 2
    assert "sine/identity" in report["cells"]
    fams = report["cells"]["sine/identity"]["families"]
    assert "lsb" in fams and "dct" in fams
    op = report["operating_point"]
    assert op["status"] == "provisional"
    assert op["n"] == 2
    assert "ab" in report
    assert "flip_default" in report["ab"]
    md = render_markdown(report)
    assert "lsb" in md
    out = write_outputs(
        report,
        json_path=tmp_path / "latest.json",
        md_path=tmp_path / "latest.md",
        operating_point_path=tmp_path / "op.json",
        write_operating_point=True,
    )
    assert Path(out["json"]).is_file()
    assert Path(out["operating_point"]).is_file()
    nested = (report.get("ab") or {}).get("nested_holdout")
    assert nested is None or "legacy_fpr" in nested


def test_bench_camera_covers(tmp_path: Path) -> None:
    proto = load_protocol()
    plates = []
    for i in range(4):
        img = synthetic_cover(96, 96, np.random.default_rng(20 + i), style="photo")
        p = tmp_path / f"cam-{i}.png"
        Image.fromarray(img).save(p)
        plates.append(img)
    loaded = load_camera_covers(tmp_path, n=4, size=64, seed=1)
    assert len(loaded) == 4
    assert loaded[0].shape == (64, 64, 3)
    report = run_bench(
        proto,
        n=4,
        size=64,
        families=["lsb", "dct"],
        attacks=["identity"],
        detectors=["chi_square", "dct"],
        covers=plates,
    )
    assert report["corpus"] == "camera"
    assert report["corpus_id"] == "camera"
    assert "camera/identity" in report["cells"]
    assert "fpr_at_locks" in report
    op = report["operating_point"]
    assert op["corpus"] == "camera"
    assert "camera" in op["id"]
    other = run_bench(
        proto,
        n=4,
        size=64,
        families=["lsb"],
        attacks=["identity"],
        detectors=["chi_square"],
        covers=plates,
        corpus_id="camera-div2k",
    )
    assert other["corpus_id"] == "camera-div2k"
    assert "camera-div2k" in other["operating_point"]["id"]
    assert other["operating_point"]["corpus"] == "camera-div2k"
    assert "Generator covers" not in (op.get("notes") or "")
    before = DEFAULT_OP.read_text(encoding="utf-8") if DEFAULT_OP.is_file() else None
    try:
        write_outputs(
            report,
            json_path=tmp_path / "camera.json",
            md_path=tmp_path / "camera.md",
            operating_point_path=DEFAULT_OP,
            write_operating_point=True,
        )
        raise AssertionError("camera bench must not write the generator OP path")
    except ValueError as exc:
        assert "generator" in str(exc).lower()
    after = DEFAULT_OP.read_text(encoding="utf-8") if DEFAULT_OP.is_file() else None
    assert after == before
    cam_op = tmp_path / "operating_point.camera.json"
    written = write_outputs(
        report,
        json_path=tmp_path / "camera.json",
        md_path=tmp_path / "camera.md",
        operating_point_path=cam_op,
        write_operating_point=True,
    )
    assert Path(written["operating_point"]).is_file()
