from veilscan.ensemble import fuse
from veilscan.types import DetectionResult, SCHEMA_VERSION


def _r(name: str, score: float, skipped: bool = False) -> DetectionResult:
    return DetectionResult(name, score, 0.9, "x", skipped=skipped, tier="deep")


def test_schema_version_is_5() -> None:
    assert SCHEMA_VERSION == 5


def test_legacy_family_hint_lsb() -> None:
    crowd = [
        _r("residual_cnn", 0.91),
        _r("fsnet_lite", 0.12),
        _r("chi_square", 0.40),
        _r("dct", 0.22),
    ]
    mix = {"mode": "legacy", "t_lsb": 0.50, "t_freq": 0.50, "t_class": 0.48, "peak": 0.4}
    r = fuse(crowd, {"residual_cnn": 1.2, "fsnet_lite": 1.2, "chi_square": 1.0, "dct": 1.0}, 0.48, (32, 32, 3), mix=mix)
    assert r.lsb_score == 0.91
    assert r.freq_score == 0.12
    js = r.to_json()
    assert js["schema_version"] == 5
    assert "jpeg_container" in js
    assert "jpeg_freq_weight" in js
    assert js.get("policy") == "generator"
    assert "jpeg_like" in js
    assert "family_hint" in js


def test_specialist_or_lsb() -> None:
    crowd = [
        _r("residual_cnn", 0.91),
        _r("fsnet_lite", 0.11),
        _r("chi_square", 0.20),
        _r("dct", 0.18),
    ]
    mix = {"mode": "specialist_or", "t_lsb": 0.50, "t_freq": 0.50, "t_class": 0.48}
    r = fuse(crowd, {"residual_cnn": 1.0, "fsnet_lite": 1.0, "chi_square": 1.0, "dct": 1.0}, 0.48, (16, 16, 3), mix=mix)
    assert r.present
    assert r.family_hint == "lsb"


def test_specialist_or_mixed() -> None:
    crowd = [
        _r("residual_cnn", 0.88),
        _r("fsnet_lite", 0.86),
        _r("chi_square", 0.30),
    ]
    mix = {"mode": "specialist_or", "t_lsb": 0.50, "t_freq": 0.50, "t_class": 0.48}
    r = fuse(crowd, {"residual_cnn": 1.0, "fsnet_lite": 1.0, "chi_square": 1.0}, 0.48, (16, 16, 3), mix=mix)
    assert r.present
    assert r.family_hint == "mixed"


def test_specialist_or_quiet_cover() -> None:
    crowd = [
        _r("residual_cnn", 0.08),
        _r("fsnet_lite", 0.05),
        _r("chi_square", 0.22),
        _r("dct", 0.19),
    ]
    mix = {"mode": "specialist_or", "t_lsb": 0.50, "t_freq": 0.50, "t_class": 0.48}
    r = fuse(crowd, {"residual_cnn": 1.0, "fsnet_lite": 1.0, "chi_square": 1.0, "dct": 1.0}, 0.48, (16, 16, 3), mix=mix)
    assert not r.present
    assert r.family_hint == "none"


def test_skipped_specialists_are_zero() -> None:
    crowd = [
        _r("residual_cnn", 0.0, skipped=True),
        _r("fsnet_lite", 0.0, skipped=True),
        _r("chi_square", 0.93),
    ]
    mix = {"mode": "specialist_or", "t_lsb": 0.50, "t_freq": 0.50, "t_class": 0.48}
    r = fuse(crowd, {"residual_cnn": 1.0, "fsnet_lite": 1.0, "chi_square": 1.0}, 0.48, (16, 16, 3), mix=mix)
    assert r.lsb_score == 0.0
    assert r.freq_score == 0.0
    assert r.present
    assert r.family_hint == "classical"


def test_jpeg_like_pulls_toward_freq() -> None:
    crowd = [
        _r("residual_cnn", 0.10),
        _r("fsnet_lite", 0.90),
        _r("chi_square", 0.70),
        _r("dct", 0.68),
    ]
    mix = {"mode": "legacy", "jpeg_like": True, "jpeg_freq_weight": 0.5, "peak": 0.4}
    r = fuse(
        crowd,
        {"residual_cnn": 1.0, "fsnet_lite": 1.2, "chi_square": 1.0, "dct": 1.0},
        0.67,
        (16, 16, 3),
        mix=mix,
    )
    mix_off = dict(mix)
    mix_off["jpeg_like"] = False
    r_off = fuse(
        crowd,
        {"residual_cnn": 1.0, "fsnet_lite": 1.2, "chi_square": 1.0, "dct": 1.0},
        0.67,
        (16, 16, 3),
        mix=mix_off,
    )
    assert r.jpeg_like is True
    assert r.score > r_off.score
    assert r.freq_score == 0.90
