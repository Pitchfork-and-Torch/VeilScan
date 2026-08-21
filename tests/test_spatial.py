import numpy as np

from veilscan.detectors.spatial import (
    ChiSquareDetector,
    RSAnalysisDetector,
    SamplePairDetector,
    rs_payload_hat,
    spa_payload_hat,
)
from veilscan.generators import embed_lsb, synthetic_cover


def _sine(seed: int = 11):
    return synthetic_cover(128, 128, np.random.default_rng(seed), style="sine")


def _pair() -> tuple[np.ndarray, np.ndarray]:
    cover = _sine(11)
    marked = embed_lsb(cover, np.random.default_rng(12), rate=0.8)
    return cover, marked


def test_chi_square_lsb() -> None:
    cover, marked = _pair()
    d = ChiSquareDetector()
    c = d.analyze(cover).score
    m = d.analyze(marked).score
    assert m > c
    assert m > 0.4


def test_chi_square_sequential_prefix() -> None:
    cover = _sine(21)
    marked = embed_lsb(cover, np.random.default_rng(22), rate=0.12, sequential=True)
    d = ChiSquareDetector()
    rc = d.analyze(cover)
    rm = d.analyze(marked)
    assert rm.extras["p_sequential"] >= rc.extras["p_sequential"] - 1e-6
    assert rm.score >= rc.score


def test_rs_lsb() -> None:
    cover, marked = _pair()
    d = RSAnalysisDetector()
    c = d.analyze(cover).score
    m = d.analyze(marked).score
    assert m >= c - 0.05
    assert m > 0.2


def test_rs_payload_rate_ladder() -> None:
    cover = _sine(31)
    mask = np.array([0, 1, 1, 0], dtype=np.int8)
    marked = embed_lsb(cover, np.random.default_rng(41), rate=1.0)
    _, m0 = rs_payload_hat(cover[..., 1], mask)
    _, m1 = rs_payload_hat(marked[..., 1], mask)
    d = RSAnalysisDetector()
    assert d.analyze(marked).score >= d.analyze(cover).score - 0.05
    assert m1["gap"] >= m0["gap"] - 0.02
    assert 0.0 <= m0["payload_hat"] <= 1.0
    assert 0.0 <= m1["payload_hat"] <= 1.0


def test_spa_lsb() -> None:
    cover, marked = _pair()
    d = SamplePairDetector()
    rc = d.analyze(cover)
    rm = d.analyze(marked)
    assert 0.0 <= rc.score <= 1.0
    assert 0.0 <= rm.score <= 1.0
    assert rm.score >= rc.score - 0.08


def test_spa_payload_rate_ladder() -> None:
    cover = synthetic_cover(192, 192, np.random.default_rng(51), style="photo")
    p0, _, n0, _ = spa_payload_hat(cover[..., 1])
    marked = embed_lsb(cover, np.random.default_rng(52), rate=1.0)
    p1, _, n1, _ = spa_payload_hat(marked[..., 1])
    assert n0 >= 20 and n1 >= 20
    # Photo covers should not already be fully equalized; payload raises the hat.
    if p0 < 0.9:
        assert p1 >= p0
    else:
        assert p1 >= 0.0
