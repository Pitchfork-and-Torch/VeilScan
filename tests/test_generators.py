import numpy as np

from veilscan.generators import FAMILIES, embed, synthetic_cover


def test_cover_shape() -> None:
    rng = np.random.default_rng(1)
    img = synthetic_cover(64, 80, rng)
    assert img.shape == (64, 80, 3)
    assert img.dtype == np.uint8


def test_dct_batch_matches_cv2() -> None:
    import cv2

    from veilscan.dsp import dct2_batch, idct2_batch

    rng = np.random.default_rng(4)
    blocks = rng.normal(0, 12, size=(3, 5, 8, 8)).astype(np.float32)
    batch = dct2_batch(blocks)
    loop = np.empty_like(blocks)
    for i in range(3):
        for j in range(5):
            loop[i, j] = cv2.dct(blocks[i, j])
    assert float(np.max(np.abs(batch - loop))) < 2e-4
    rec = idct2_batch(batch)
    assert float(np.max(np.abs(rec - blocks))) < 2e-4


def test_embed_dct_spread_fast_enough() -> None:
    cover = synthetic_cover(128, 128, np.random.default_rng(5), style="photo")
    import time

    t0 = time.perf_counter()
    a = embed(cover, "dct", seed=9)
    b = embed(cover, "spread", seed=10)
    elapsed = time.perf_counter() - t0
    assert a.shape == cover.shape and b.shape == cover.shape
    assert elapsed < 1.5


def test_all_families_preserve_shape() -> None:
    rng = np.random.default_rng(2)
    cover = synthetic_cover(96, 96, rng)
    for fam in FAMILIES:
        out = embed(cover, fam, seed=3)
        assert out.shape == cover.shape
        assert out.dtype == np.uint8
        assert not np.array_equal(out, cover)
