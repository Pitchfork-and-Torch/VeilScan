from pathlib import Path

import numpy as np

from veilscan import __version__, analyze, analyze_images
from veilscan.calibrate import apply_affine, fit_calibration, identity_calibration, load_calibration, save_calibration
from veilscan.eval.harness import run_robustness
from veilscan.generators import embed, synthetic_cover
from veilscan.types import SCHEMA_VERSION


def test_version() -> None:
    assert __version__ == "2.6.0"


def test_schema_version_json() -> None:
    rng = np.random.default_rng(0)
    img = synthetic_cover(64, 64, rng, style="sine")
    r = analyze(img, detectors=["chi_square"])
    js = r.to_json()
    assert js["schema_version"] == SCHEMA_VERSION
    assert "family_hint" in js
    assert "lsb_score" in js
    assert "freq_score" in js


def test_camera_sidecar_omitted_and_attached(monkeypatch) -> None:
    rng = np.random.default_rng(0)
    img = synthetic_cover(64, 64, rng, style="sine")
    monkeypatch.setattr("veilscan.engine.load_camera_operating_point", lambda path=None: None)
    assert "camera" not in analyze(img, detectors=["chi_square"]).to_json()
    monkeypatch.setattr(
        "veilscan.engine.load_camera_operating_point",
        lambda path=None: {
            "id": "op-cam-test",
            "status": "provisional",
            "threshold": 0.99,
            "fpr_est": 0.05,
            "n": 4,
        },
    )
    js = analyze(img, detectors=["chi_square"]).to_json()
    assert js["camera"]["operating_point_id"] == "op-cam-test"
    assert js["camera"]["status"] == "provisional"
    assert "present" in js["camera"]
    assert js["present"] is False or js["present"] is True


def test_camera_sidecar_file_does_not_replace_generator() -> None:
    from veilscan.config import load_camera_operating_point, load_operating_point

    gen = load_operating_point()
    cam = load_camera_operating_point()
    assert gen is not None
    assert gen.get("id") == "op-v0.4.0-locked-n50"
    assert cam is not None
    cam_id = str(cam.get("id") or "")
    assert cam_id.startswith("op-v1.") and "camera" in cam_id and "div2k" not in cam_id
    assert cam.get("status") == "locked"
    assert cam.get("id") != gen.get("id")
    assert float(cam["threshold"]) != float(gen["threshold"])


def test_present_policy_default_is_generator(tmp_path) -> None:
    from veilscan.api import analyze_path
    from veilscan.engine import apply_present_policy
    from veilscan.generators import synthetic_cover
    from PIL import Image

    img = synthetic_cover(64, 64, np.random.default_rng(9), style="sine")
    p = tmp_path / "x.png"
    Image.fromarray(img).save(p)
    r = analyze_path(p, detectors=["chi_square"])
    assert r.policy == "generator"
    cam = {"operating_point_id": "op-cam-test", "status": "locked", "threshold": 0.99, "present": False, "n": 4}
    r.camera = dict(cam)
    r.score = 0.80
    r.present = True
    r.threshold = 0.67
    r.present = True
    r.threshold = 0.67
    gen = apply_present_policy(r, "generator")
    assert gen.present is True
    assert gen.threshold == 0.67
    r.present = True
    r.threshold = 0.67
    cam_p = apply_present_policy(r, "camera")
    assert cam_p.present is False
    assert cam_p.threshold == 0.99
    r.present = True
    r.threshold = 0.67
    both = apply_present_policy(r, "both")
    assert both.policy == "both"
    assert both.present is True
    assert both.camera["present"] is False


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
