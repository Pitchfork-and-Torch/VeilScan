"""Deep detectors. Silent unless a checkpoint file exists."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult

_SIZE = 128


def _checkpoint_dir(context: AnalyzeContext | None) -> Path:
    if context and context.checkpoint_dir:
        return Path(context.checkpoint_dir)
    return Path(__file__).resolve().parents[3] / "checkpoints"


def native_windows(image: np.ndarray, size: int = _SIZE) -> list[np.ndarray]:
    """128px windows at native scale. Downscaling 256+ DCT marks to 128 is chance.

    Exact size: one window, no resample. Smaller: area-resize. Larger: four
    corners plus center. Mean of those windows is the detector score.
    """
    import cv2

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("RGB required")
    rgb = np.ascontiguousarray(image[..., :3])
    h, w = rgb.shape[:2]
    if h < size or w < size:
        return [cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)]
    if h == size and w == size:
        return [rgb]
    coords = [
        (0, 0),
        (0, w - size),
        (h - size, 0),
        (h - size, w - size),
        ((h - size) // 2, (w - size) // 2),
    ]
    out: list[np.ndarray] = []
    seen: set[tuple[int, int]] = set()
    for y, x in coords:
        y_i = int(max(0, y))
        x_i = int(max(0, x))
        key = (y_i, x_i)
        if key in seen:
            continue
        seen.add(key)
        tile = rgb[y_i : y_i + size, x_i : x_i + size]
        if tile.shape[0] == size and tile.shape[1] == size:
            out.append(tile)
    if out:
        return out
    return [cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)]


def _infer(model, image: np.ndarray, device: str) -> tuple[float, int]:
    import torch

    model.eval()
    windows = native_windows(image)
    batch = torch.stack(
        [torch.from_numpy(w.astype(np.float32) / 255.0).permute(2, 0, 1) for w in windows]
    )
    with torch.no_grad():
        logit = model(batch.to(device))
        prob = torch.sigmoid(logit.reshape(-1))
        return float(prob.mean().cpu()), int(prob.numel())


class ResidualCNNDetector(BaseDetector):
    name = "residual_cnn"
    tier = "deep"
    requires_checkpoint = True

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        ckpt = _checkpoint_dir(context) / "residual_cnn.pt"
        if not ckpt.is_file():
            return self.skip("no checkpoints/residual_cnn.pt (train with scripts/train_lite.py)")
        try:
            import torch
            from veilscan.models.residual_cnn import ResidualCNN
        except Exception as e:
            return self.skip(f"torch unavailable: {e}")
        device = (context.device if context else "cpu") or "cpu"
        model = ResidualCNN()
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        p, n_win = _infer(model, image, device)
        expl = f"Residual CNN (YeNet/SRNet-lite) p={p:.3f} from {ckpt.name} windows={n_win}."
        return DetectionResult(
            self.name,
            p,
            0.7,
            expl,
            extras={"ckpt": str(ckpt), "n_windows": n_win},
            tier=self.tier,
        ).clamp()


class FSNetLiteDetector(BaseDetector):
    name = "fsnet_lite"
    tier = "deep"
    requires_checkpoint = True

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        ckpt = _checkpoint_dir(context) / "fsnet_lite.pt"
        if not ckpt.is_file():
            return self.skip("no checkpoints/fsnet_lite.pt (train with scripts/train_lite.py)")
        try:
            import torch
            from veilscan.models.fsnet import FSNetLite
        except Exception as e:
            return self.skip(f"torch unavailable: {e}")
        device = (context.device if context else "cpu") or "cpu"
        model = FSNetLite()
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        p, n_win = _infer(model, image, device)
        expl = f"FSNet-lite (ASPM+DMSA) p={p:.3f} from {ckpt.name} windows={n_win}."
        return DetectionResult(
            self.name,
            p,
            0.75,
            expl,
            extras={"ckpt": str(ckpt), "n_windows": n_win},
            tier=self.tier,
        ).clamp()


def register_deep() -> None:
    register(ResidualCNNDetector())
    register(FSNetLiteDetector())
