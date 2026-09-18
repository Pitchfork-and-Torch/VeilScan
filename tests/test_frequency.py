import numpy as np

from veilscan.detectors.frequency import DCTDetector, DWTDetector, TreeRingSpectralDetector
from veilscan.generators import embed_dct, embed_dwt, embed_tree_ring, synthetic_cover


def test_dct_detector() -> None:
    rng = np.random.default_rng(21)
    cover = synthetic_cover(128, 128, rng)
    marked = embed_dct(cover, np.random.default_rng(22), amp=12.0)
    d = DCTDetector()
    rm = d.analyze(marked)
    rc = d.analyze(cover)
    assert rm.score > rc.score
    assert rm.extras["pair_34_43"] > rc.extras["pair_34_43"]


def test_dwt_detector() -> None:
    rng = np.random.default_rng(23)
    cover = synthetic_cover(128, 128, rng)
    marked = embed_dwt(cover, np.random.default_rng(24), strength=0.2)
    d = DWTDetector()
    assert d.analyze(marked).score > d.analyze(cover).score


def test_tree_ring_detector() -> None:
    rng = np.random.default_rng(25)
    cover = synthetic_cover(128, 128, rng)
    marked = embed_tree_ring(cover, np.random.default_rng(26), strength=0.35)
    d = TreeRingSpectralDetector()
    rm = d.analyze(marked)
    rc = d.analyze(cover)
    assert rm.score >= rc.score - 0.05
    if not rm.skipped:
        assert "scale_scores" in rm.extras
