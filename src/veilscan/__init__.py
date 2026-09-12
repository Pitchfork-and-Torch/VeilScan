"""VeilScan: watermark presence, keyless decode, and forensic hunt extract."""

from veilscan.api import analyze, analyze_images, analyze_path, list_detectors
from veilscan.decode import decode_bytes, decode_path
from veilscan.hunt import hunt_bytes, hunt_path
from veilscan.types import DetectionResult, EnsembleResult

__version__ = "2.2.0"
__all__ = [
    "analyze",
    "analyze_images",
    "analyze_path",
    "decode_bytes",
    "decode_path",
    "hunt_bytes",
    "hunt_path",
    "list_detectors",
    "DetectionResult",
    "EnsembleResult",
    "__version__",
]
