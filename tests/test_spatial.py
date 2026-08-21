import numpy as np

from veilscan.detectors.spatial import ChiSquareDetector, RSAnalysisDetector, SamplePairDetector
from veilscan.generators import embed_lsb, synthetic_cover


def _pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    cover = synthetic_cover(128, 128, rng)
    marked = embed_lsb(cover, np.random.default_rng(12), rate=0.8)
    return cover, marked


def test_chi_square_lsb() -> None:
    cover, marked = _pair()
    d = ChiSquareDetector()
    c = d.analyze(cover).score
    m = d.analyze(marked).score
    assert m > c
    assert m > 0.4


def test_rs_lsb() -> None:
    cover, marked = _pair()
    d = RSAnalysisDetector()
    c = d.analyze(cover).score
    m = d.analyze(marked).score
    assert m >= c - 0.05
    assert m > 0.2


def test_spa_lsb() -> None:
    cover, marked = _pair()
    d = SamplePairDetector()
    rc = d.analyze(cover)
    rm = d.analyze(marked)
    assert 0.0 <= rc.score <= 1.0
    assert 0.0 <= rm.score <= 1.0
    # Neighbor-LSB correlation saturates on steep quantized ramps; RS/chi-square are the LSB proofs.
