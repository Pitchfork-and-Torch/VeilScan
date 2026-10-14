"""Detector plugin base."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from veilscan.types import AnalyzeContext, DetectionResult, Tier


class BaseDetector(ABC):
    name: str
    tier: Tier = "fast"
    requires_reference: bool = False
    requires_checkpoint: bool = False

    @abstractmethod
    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        raise NotImplementedError

    def skip(self, reason: str) -> DetectionResult:
        return DetectionResult(
            detector=self.name,
            score=0.0,
            confidence=0.0,
            explanation=reason,
            skipped=True,
            tier=self.tier,
        ).clamp()
