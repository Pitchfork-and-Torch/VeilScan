import importlib.util
from pathlib import Path

import numpy as np
import torch

from veilscan.generators import FAMILIES, synthetic_cover
from veilscan.models.fsnet import FSNetLite


def _train_mod():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_lite.py"
    spec = importlib.util.spec_from_file_location("veilscan_train_lite", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_holdout_excludes_family() -> None:
    mod = _train_mod()
    ds = mod.SyntheticWM(8, seed=0, holdout="lsb", jpeg_prob=0.0)
    assert "lsb" not in ds.families
    assert set(ds.families) == set(FAMILIES) - {"lsb"}
    x, y = ds[0]
    assert x.shape[0] == 3
    assert x.shape[1] == x.shape[2]
    assert y.ndim == 0


def test_photo_cover_shape() -> None:
    img = synthetic_cover(96, 80, np.random.default_rng(0), style="photo")
    assert img.shape == (96, 80, 3)
    assert img.dtype == np.uint8


def test_fsnet_rgb_aspm_forward() -> None:
    m = FSNetLite()
    m.eval()
    x = torch.zeros(2, 3, 64, 64)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2,)
