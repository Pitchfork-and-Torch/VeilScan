from pathlib import Path

from PIL import Image

from veilscan.eval.bench import load_camera_covers, load_protocol, render_markdown, run_bench, write_outputs
from veilscan.eval.metrics import cut_at_fpr, fpr_at
from veilscan.generators import synthetic_cover
import numpy as np


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
    assert "camera/identity" in report["cells"]
