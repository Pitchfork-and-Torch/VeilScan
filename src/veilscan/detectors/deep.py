"""Deep detectors. Silent unless a checkpoint file exists."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.dsp import sigmoid
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult

_SIZE = 128


def _checkpoint_dir(context: AnalyzeContext | None) -> Path:
    if context and context.checkpoint_dir:
        return Path(context.checkpoint_dir)
    return Path(__file__).resolve().parents[3] / "checkpoints"


def _load_rgb_tensor(image: np.ndarray, device: str):
    import torch
    import cv2

    rgb = image
    if rgb.shape[2] != 3:
        raise ValueError("RGB required")
    resized = cv2.resize(rgb, (_SIZE, _SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return t.to(device)


def _infer(model, image: np.ndarray, device: str) -> float:
    import torch

    model.eval()
    x = _load_rgb_tensor(image, device)
    with torch.no_grad():
        logit = model(x)
        if logit.ndim > 0:
            logit = logit.reshape(-1)[0]
        return float(torch.sigmoid(logit).cpu())


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
        p = _infer(model, image, device)
        expl = f"Residual CNN (YeNet/SRNet-lite) p={p:.3f} from {ckpt.name}."
        return DetectionResult(self.name, p, 0.7, expl, extras={"ckpt": str(ckpt)}, tier=self.tier).clamp()


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
        p = _infer(model, image, device)
        expl = f"FSNet-lite (ASPM+DMSA) p={p:.3f} from {ckpt.name}."
        return DetectionResult(self.name, p, 0.75, expl, extras={"ckpt": str(ckpt)}, tier=self.tier).clamp()


def register_deep() -> None:
    register(ResidualCNNDetector())
    register(FSNetLiteDetector())
