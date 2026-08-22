import numpy as np

from veilscan.dsp import haar_dwt2, haar_idwt2, jpeg_blockiness, jpeg_like
from veilscan.eval.robustness import apply_attack
from veilscan.generators import synthetic_cover


def test_haar_roundtrip() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 32))
    rec = haar_idwt2(*haar_dwt2(x))
    assert rec.shape == x.shape
    assert np.mean((rec - x) ** 2) < 1e-8


def test_jpeg_like_after_roundtrip() -> None:
    cover = synthetic_cover(96, 96, np.random.default_rng(3), style="sine")
    jpeg = apply_attack(cover, "jpeg_70", np.random.default_rng(4))
    assert jpeg_blockiness(jpeg) > jpeg_blockiness(cover)
    assert jpeg_like(jpeg)
    assert not jpeg_like(cover)
