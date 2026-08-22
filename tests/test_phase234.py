from pathlib import Path

import numpy as np

from veilscan import __version__, analyze, analyze_images
from veilscan.calibrate import apply_affine, fit_calibration, identity_calibration, load_calibration, save_calibration
from veilscan.eval.harness import run_robustness
from veilscan.generators import embed, synthetic_cover
from veilscan.types import SCHEMA_VERSION


def test_version() -> None:
    assert __version__ == "0.4.1"


def test_schema_version_json() -> None:
    rng = np.random.default_rng(0)
    img = synthetic_cover(64, 64, rng, style="sine")
    r = analyze(img, detectors=["chi_square"])
    js = r.to_json()
    assert js["schema_version"] == SCHEMA_VERSION
    assert "family_hint" in js
    assert "lsb_score" in js
    assert "freq_score" in js


def test_identity_calibration_noop() -> None:
    cal = identity_calibration()
    assert apply_affine(cal["ensemble"], 0.4) == apply_affine({"a": 1.0, "b": 0.0}, 0.4)


def test_fit_calibration_writes(tmp_path: Path) -> None:
    data = fit_calibration(n=2, size=64, families=["lsb", "dct"], detectors=["chi_square", "dct"], seed=1)
    assert "ensemble" in data and "a" in data["ensemble"]
    dest = tmp_path / "cal.json"
    save_calibration(data, dest)
    loaded = load_calibration(dest)
    assert loaded["fit"]["n"] == 2


def test_analyze_images_batch() -> None:
    rng = np.random.default_rng(2)
    imgs = [synthetic_cover(48, 48, np.random.default_rng(10 + i), style="sine") for i in range(2)]
    out = analyze_images(imgs, detectors=["chi_square", "dct"])
    assert len(out) == 2


def test_inversion_and_gs_skip() -> None:
    img = synthetic_cover(48, 48, np.random.default_rng(3), style="sine")
    r = analyze(img, detectors=["tree_ring_inversion", "gaussian_shading"])
    assert all(d.skipped for d in r.detectors)


def test_tree_ring_multiscale_extras() -> None:
    rng = np.random.default_rng(4)
    cover = synthetic_cover(96, 96, rng, style="sine")
    marked = embed(cover, "tree_ring", seed=5)
    r = analyze(marked, detectors=["tree_ring_spectral"])
    d = r.detectors[0]
    if not d.skipped:
        assert "scale_scores" in d.extras


def test_robustness_retention_keys() -> None:
    from veilscan.config import VeilConfig

    cfg = VeilConfig.load()
    report = run_robustness(
        n=1,
        size=64,
        family="dct",
        seed=6,
        attacks=["identity", "jpeg_70"],
        detectors=["dct"],
        cfg=cfg,
    )
    assert "identity" in report["attacks"]
    assert "retention" in report["attacks"]["identity"]


def test_wmd_prune_tiny() -> None:
    from veilscan.detectors.blackbox import wmd_prune_scan

    rng = np.random.default_rng(7)
    refs = [synthetic_cover(64, 64, np.random.default_rng(20 + i), style="sine") for i in range(2)]
    covers = [synthetic_cover(64, 64, np.random.default_rng(30 + i), style="sine") for i in range(2)]
    marked = [embed(covers[0], "dct", seed=8), embed(covers[1], "lsb", seed=9)]
    report = wmd_prune_scan(marked + covers, refs, rounds=1, steps=2, keep=0.75, device="cpu")
    assert report["n_suspects"] == 4
    assert len(report["items"]) == 4


def test_onnx_export(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("onnx")
    from veilscan.export import export_residual_cnn

    out = tmp_path / "rcnn.onnx"
    export_residual_cnn(out, None, size=32)
    assert out.is_file() and out.stat().st_size > 100
