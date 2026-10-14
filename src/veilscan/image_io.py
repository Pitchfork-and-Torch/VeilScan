"""Image load / save / tiling."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    return load_rgb_bytes(path.read_bytes())


def load_rgb_bytes(data: bytes) -> np.ndarray:
    img = Image.open(BytesIO(data))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    elif img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV required for JPEG probe")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def iter_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
    return files


def tiles(rgb: np.ndarray, tile: int, overlap: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    h, w = rgb.shape[:2]
    if max(h, w) <= tile:
        return [(0, 0, h, w, rgb)]
    step = max(1, tile - overlap)
    out = []
    y = 0
    while y < h:
        x = 0
        y2 = min(h, y + tile)
        y1 = max(0, y2 - tile) if y2 - y < tile and y != 0 else y
        while x < w:
            x2 = min(w, x + tile)
            x1 = max(0, x2 - tile) if x2 - x < tile and x != 0 else x
            out.append((y1, x1, y2, x2, rgb[y1:y2, x1:x2]))
            if x2 >= w:
                break
            x += step
        if y2 >= h:
            break
        y += step
    return out
