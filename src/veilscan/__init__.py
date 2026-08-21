"""VeilScan: agnostic invisible-watermark presence detection plus keyless text decode."""

from veilscan.api import analyze, analyze_images, analyze_path, list_detectors
from veilscan.decode import decode_bytes, decode_path
from veilscan.types import DetectionResult, EnsembleResult

__version__ = "0.3.2"
__all__ = [
    "analyze",
    "analyze_images",
    "analyze_path",
    "decode_bytes",
    "decode_path",
    "list_detectors",
    "DetectionResult",
    "EnsembleResult",
    "__version__",
]
