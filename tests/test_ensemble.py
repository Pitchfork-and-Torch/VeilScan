import numpy as np

from veilscan.api import analyze, list_detectors
from veilscan.config import VeilConfig
from veilscan.generators import embed, synthetic_cover


def test_list_detectors_has_core() -> None:
    names = {d["name"] for d in list_detectors()}
    for n in (
        "chi_square",
        "rs_analysis",
        "dct",
        "dwt",
        "srm",
        "fsnet_lite",
        "wmd",
        "jpeg_ela",
        "tree_ring_inversion",
        "gaussian_shading",
    ):
        assert n in names


def test_ensemble_lsb_higher() -> None:
    from veilscan.generators import embed_lsb

    cfg = VeilConfig.load()
    cfg.tier = "fast"
    names = ["chi_square", "rs_analysis", "sample_pairs"]
    dc, dm = [], []
    for i in range(4):
        cover = synthetic_cover(128, 128, np.random.default_rng(40 + i))
        marked = embed_lsb(cover, np.random.default_rng(90 + i), rate=0.85)
        dc.append(analyze(cover, config=cfg, detectors=names).score)
        dm.append(analyze(marked, config=cfg, detectors=names).score)
    assert float(np.mean(dm)) > float(np.mean(dc)) + 0.02


def test_deep_skipped_without_ckpt() -> None:
    rng = np.random.default_rng(33)
    cover = synthetic_cover(64, 64, rng)
    r = analyze(cover, detectors=["residual_cnn", "fsnet_lite"])
    assert all(d.skipped for d in r.detectors)


def test_fuse_peak_not_drowned() -> None:
    from veilscan.ensemble import fuse
    from veilscan.types import DetectionResult

    crowd = [
        DetectionResult("chi_square", 0.92, 0.85, "x", tier="fast"),
        DetectionResult("rs_analysis", 0.71, 0.82, "x", tier="fast"),
        DetectionResult("dct", 0.51, 0.7, "x", tier="fast"),
        DetectionResult("dwt", 0.48, 0.7, "x", tier="fast"),
        DetectionResult("srm", 0.50, 0.7, "x", tier="fast"),
        DetectionResult("dft", 0.49, 0.65, "x", tier="fast"),
        DetectionResult("color_spaces", 0.47, 0.55, "x", tier="residual"),
    ]
    r = fuse(crowd, {"chi_square": 1.0, "rs_analysis": 1.0, "dct": 1.0, "dwt": 1.0, "srm": 0.9, "dft": 0.7, "color_spaces": 0.5}, 0.55, (32, 32, 3))
    assert r.score > 0.62
    assert r.present


def test_wmd_skipped_without_reference() -> None:
    rng = np.random.default_rng(34)
    cover = synthetic_cover(64, 64, rng)
    r = analyze(cover, detectors=["wmd"])
    assert r.detectors[0].skipped
