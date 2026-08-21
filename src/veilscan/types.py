"""Shared result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np

SCHEMA_VERSION = 1
Tier = Literal["fast", "frequency", "residual", "deep", "blackbox", "foundation"]


@dataclass
class DetectionResult:
    detector: str
    score: float
    confidence: float
    explanation: str
    heatmap: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    tier: str = "fast"

    def clamp(self) -> DetectionResult:
        self.score = float(np.clip(self.score, 0.0, 1.0))
        self.confidence = float(np.clip(self.confidence, 0.0, 1.0))
        return self

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        hm = d.pop("heatmap")
        d["has_heatmap"] = hm is not None
        d["score"] = round(float(self.score), 6)
        d["confidence"] = round(float(self.confidence), 6)
        extras = {}
        for k, v in self.extras.items():
            if isinstance(v, (np.floating, np.integer)):
                extras[k] = float(v)
            elif isinstance(v, np.ndarray):
                extras[k] = v.tolist()
            else:
                extras[k] = v
        d["extras"] = extras
        return d


@dataclass
class EnsembleResult:
    present: bool
    score: float
    confidence: float
    uncertainty: float
    threshold: float
    explanation: str
    detectors: list[DetectionResult]
    heatmap: np.ndarray | None = None
    image_shape: tuple[int, ...] = ()
    active: int = 0
    skipped: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "present": self.present,
            "score": round(float(self.score), 6),
            "confidence": round(float(self.confidence), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "threshold": self.threshold,
            "explanation": self.explanation,
            "active": self.active,
            "skipped": self.skipped,
            "image_shape": list(self.image_shape),
            "has_heatmap": self.heatmap is not None,
            "detectors": [d.to_json() for d in self.detectors],
        }


@dataclass
class AnalyzeContext:
    device: str = "cpu"
    reference_images: list[np.ndarray] | None = None
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
