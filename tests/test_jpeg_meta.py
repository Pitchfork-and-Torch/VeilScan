from io import BytesIO

import numpy as np
from PIL import Image

from veilscan.api import analyze, analyze_path
from veilscan.generators import synthetic_cover
from veilscan.jpeg_meta import inspect_jpeg, is_jpeg, jpeg_freq_weight
from veilscan.types import SCHEMA_VERSION


def _jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=int(quality), subsampling=0)
    return buf.getvalue()


def test_png_is_not_jpeg() -> None:
    cover = synthetic_cover(64, 64, np.random.default_rng(2), style="sine")
    buf = BytesIO()
    Image.fromarray(cover).save(buf, format="PNG")
    data = buf.getvalue()
    info = inspect_jpeg(data)
    assert not is_jpeg(data)
    assert info["container"] is False
    assert info["quality_est"] is None


def test_jpeg_container_and_quality_order() -> None:
    cover = synthetic_cover(96, 96, np.random.default_rng(3), style="photo")
    q90 = inspect_jpeg(_jpeg_bytes(cover, 90))
    q50 = inspect_jpeg(_jpeg_bytes(cover, 50))
    assert q90["container"] is True
    assert q50["container"] is True
    assert q90["quality_est"] is not None and q50["quality_est"] is not None
    assert q90["quality_est"] > q50["quality_est"]
    assert q90["subsampling"] in {"4:4:4", "4:2:2", "4:2:0"} or q90["subsampling"] is not None


def test_schema_5_and_analyze_path_jpeg(tmp_path) -> None:
    assert SCHEMA_VERSION == 5
    cover = synthetic_cover(64, 64, np.random.default_rng(4), style="sine")
    path = tmp_path / "plate.jpg"
    path.write_bytes(_jpeg_bytes(cover, 70))
    r = analyze_path(path, detectors=["chi_square"])
    js = r.to_json()
    assert js["schema_version"] == 5
    assert js["jpeg_container"] is True
    assert js["jpeg_like"] is True
    assert js["jpeg_quality_est"] is not None
    assert 0.4 <= float(js["jpeg_freq_weight"]) <= 0.55
    png = tmp_path / "plate.png"
    Image.fromarray(cover).save(png)
    p = analyze_path(png, detectors=["chi_square"])
    assert p.jpeg_container is False
    assert p.jpeg_freq_weight == 0.0


def test_jpeg_freq_weight_monotonic() -> None:
    assert jpeg_freq_weight(None, jpeg_like=False) == 0.0
    w50 = jpeg_freq_weight(50, jpeg_like=True)
    w70 = jpeg_freq_weight(70, jpeg_like=True)
    w90 = jpeg_freq_weight(90, jpeg_like=True)
    assert w50 == 0.70
    assert w90 == 0.25
    assert w50 > w70 > w90
    assert jpeg_freq_weight(None, jpeg_like=True, blockiness=1.25) == 0.70
    assert jpeg_freq_weight(None, jpeg_like=True, blockiness=1.12) == 0.50
    assert jpeg_freq_weight(None, jpeg_like=True, blockiness=1.00) == 0.25


def test_array_only_stays_pixel_jpeg_like() -> None:
    cover = synthetic_cover(64, 64, np.random.default_rng(5), style="sine")
    r = analyze(cover, detectors=["chi_square"])
    assert r.jpeg_container is False
    assert r.jpeg_like is False
