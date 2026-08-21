import numpy as np

from veilscan.config import VeilConfig
from veilscan.eval.harness import run_loao, run_synthetic
from veilscan.generators import embed, synthetic_cover
from veilscan.api import analyze, list_detectors


def test_per_detector_table_shape() -> None:
    cfg = VeilConfig.load()
    names = ["chi_square", "dct"]
    report = run_synthetic(n=2, size=64, families=["lsb", "dct"], seed=3, cfg=cfg, detectors=names, per_detector=True)
    assert "detectors" in report
    assert "chi_square" in report["detectors"]
    assert "lsb" in report["detectors"]["chi_square"]
    assert 0.0 <= report["detectors"]["chi_square"]["lsb"]["auc"] <= 1.0


def test_loao_skeleton() -> None:
    cfg = VeilConfig.load()
    report = run_loao(n=2, size=64, families=["lsb", "dct"], seed=4, cfg=cfg, detectors=["chi_square", "dct"])
    assert report["deep_loao"] == "not_run"
    assert "lsb" in report["held_out_ensemble"]
    assert "dct" in report["detectors"]


def test_peak_ok_from_yaml() -> None:
    cfg = VeilConfig.load()
    assert "chi_square" in cfg.peak_ok
    assert "residual_cnn" in cfg.peak_ok
    assert "dct" in cfg.peak_ok
    assert "sample_pairs" not in cfg.peak_ok
    assert "block_multiscale" not in cfg.peak_ok


def test_jpeg_ela_registered() -> None:
    names = {d["name"] for d in list_detectors()}
    assert "jpeg_ela" in names


def test_jpeg_ela_runs() -> None:
    rng = np.random.default_rng(5)
    cover = synthetic_cover(64, 64, rng)
    marked = embed(cover, "dct", seed=6)
    r = analyze(cover, detectors=["jpeg_ela"])
    m = analyze(marked, detectors=["jpeg_ela"])
    assert not r.detectors[0].skipped
    assert 0.0 <= r.detectors[0].score <= 1.0
    assert 0.0 <= m.detectors[0].score <= 1.0
