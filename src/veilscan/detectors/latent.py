"""Gated latent-family heads. Skip unless extras exist. Never recover seeds."""

from __future__ import annotations

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult


class TreeRingInversionDetector(BaseDetector):
    name = "tree_ring_inversion"
    tier = "deep"
    requires_checkpoint = True

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        try:
            import diffusers  # noqa: F401
        except Exception:
            return self.skip("DDIM inversion needs the optional diffusers package (not installed)")
        ckpt = None
        if context and context.extra:
            ckpt = context.extra.get("sd_checkpoint")
        if not ckpt:
            return self.skip("no local Stable Diffusion checkpoint configured (opt-in only)")
        return self.skip("inversion backend not wired for this checkpoint path")


class GaussianShadingDetector(BaseDetector):
    name = "gaussian_shading"
    tier = "deep"

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        return self.skip(
            "Gaussian Shading is latent-seed watermarking; pixel-only presence is not claimed. No seed recovery."
        )


def register_latent() -> None:
    register(TreeRingInversionDetector())
    register(GaussianShadingDetector())
