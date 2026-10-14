import numpy as np

from veilscan.detectors.deep import native_windows
from veilscan.generators import embed, synthetic_cover


def test_native_windows_exact_128_is_one() -> None:
    rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    wins = native_windows(rgb)
    assert len(wins) == 1
    assert wins[0].shape == (128, 128, 3)
    assert wins[0] is rgb or np.array_equal(wins[0], rgb)


def test_native_windows_small_resizes() -> None:
    rgb = np.zeros((64, 80, 3), dtype=np.uint8)
    wins = native_windows(rgb)
    assert len(wins) == 1
    assert wins[0].shape == (128, 128, 3)


def test_native_windows_256_is_five_unique() -> None:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[0, 0] = (1, 0, 0)
    rgb[0, 255] = (2, 0, 0)
    rgb[255, 0] = (3, 0, 0)
    rgb[255, 255] = (4, 0, 0)
    rgb[128, 128] = (5, 0, 0)
    wins = native_windows(rgb)
    assert len(wins) == 5
    assert all(w.shape == (128, 128, 3) for w in wins)
    keys = {(int(w[0, 0, 0]), int(w[0, -1, 0]), int(w[-1, 0, 0]), int(w[-1, -1, 0])) for w in wins}
    assert len(keys) == 5


def test_fsnet_256_dct_beats_cover() -> None:
    from pathlib import Path

    from veilscan.api import analyze

    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "fsnet_lite.pt"
    if not ckpt.is_file():
        return
    rng = np.random.default_rng(20260822)
    cover = synthetic_cover(256, 256, rng, style="photo")
    marked = embed(cover, "dct", seed=20260822)
    dc = analyze(cover, detectors=["fsnet_lite"])
    dm = analyze(marked, detectors=["fsnet_lite"])
    assert not dc.detectors[0].skipped
    assert dc.detectors[0].extras.get("n_windows") == 5
    assert dm.score > dc.score + 0.3
