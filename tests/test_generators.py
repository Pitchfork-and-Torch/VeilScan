import numpy as np

from veilscan.generators import FAMILIES, embed, synthetic_cover


def test_cover_shape() -> None:
    rng = np.random.default_rng(1)
    img = synthetic_cover(64, 80, rng)
    assert img.shape == (64, 80, 3)
    assert img.dtype == np.uint8


def test_all_families_preserve_shape() -> None:
    rng = np.random.default_rng(2)
    cover = synthetic_cover(96, 96, rng)
    for fam in FAMILIES:
        out = embed(cover, fam, seed=3)
        assert out.shape == cover.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, cover)
