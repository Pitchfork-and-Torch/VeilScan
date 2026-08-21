"""VeilScan: agnostic invisible-watermark presence detection."""

from veilscan.api import analyze, analyze_path, list_detectors
from veilscan.types import DetectionResult, EnsembleResult

__version__ = "0.1.0"
__all__ = ["analyze", "analyze_path", "list_detectors", "DetectionResult", "EnsembleResult", "__version__"]
