"""Shared result schemas."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

import numpy as np

SCHEMA_VERSION = 5
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
    family_hint: str = "none"
    lsb_score: float = 0.0
    freq_score: float = 0.0
    class_score: float = 0.0
    operating_point_id: str | None = None
    camera: dict[str, Any] | None = None
    jpeg_like: bool = False
    jpeg_blockiness: float = 0.0
    jpeg_container: bool = False
    jpeg_quality_est: int | None = None
    jpeg_subsampling: str | None = None
    jpeg_freq_weight: float = 0.0
    jpeg_luma_q_34: int | None = None
    jpeg_luma_q_43: int | None = None
    jpeg_dct_pair_34_43: float | None = None
    policy: str = "generator"

    def to_json(self) -> dict[str, Any]:
        d = {
            "schema_version": SCHEMA_VERSION,
            "present": self.present,
            "score": round(float(self.score), 6),
            "confidence": round(float(self.confidence), 6),
            "uncertainty": round(float(self.uncertainty), 6),
            "threshold": self.threshold,
            "family_hint": self.family_hint,
            "lsb_score": round(float(self.lsb_score), 6),
            "freq_score": round(float(self.freq_score), 6),
            "class_score": round(float(self.class_score), 6),
            "operating_point_id": self.operating_point_id,
            "jpeg_like": bool(self.jpeg_like),
            "jpeg_blockiness": round(float(self.jpeg_blockiness), 6),
            "jpeg_container": bool(self.jpeg_container),
            "jpeg_quality_est": self.jpeg_quality_est,
            "jpeg_subsampling": self.jpeg_subsampling,
            "jpeg_freq_weight": round(float(self.jpeg_freq_weight), 4),
            "jpeg_luma_q_34": self.jpeg_luma_q_34,
            "jpeg_luma_q_43": self.jpeg_luma_q_43,
            "jpeg_dct_pair_34_43": (
                None
                if self.jpeg_dct_pair_34_43 is None
                else round(float(self.jpeg_dct_pair_34_43), 6)
            ),
            "policy": self.policy,
            "explanation": self.explanation,
            "active": self.active,
            "skipped": self.skipped,
            "image_shape": list(self.image_shape),
            "has_heatmap": self.heatmap is not None,
            "detectors": [d.to_json() for d in self.detectors],
        }
        if self.camera:
            d["camera"] = self.camera
        return d


@dataclass
class AnalyzeContext:
    device: str = "cpu"
    reference_images: list[np.ndarray] | None = None
    checkpoint_dir: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
