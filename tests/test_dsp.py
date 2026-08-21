import numpy as np

from veilscan.dsp import haar_dwt2, haar_idwt2


def test_haar_roundtrip() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 32))
    rec = haar_idwt2(*haar_dwt2(x))
    assert rec.shape == x.shape
    assert np.mean((rec - x) ** 2) < 1e-8
