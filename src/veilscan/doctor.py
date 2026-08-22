"""Local stack check. No network. No Gradio."""

from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from veilscan import __version__
from veilscan.config import VeilConfig, load_operating_point, resolve_device

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ckpt_dir(cfg: VeilConfig) -> Path:
    if cfg.checkpoint_dir:
        return Path(cfg.checkpoint_dir)
    return ROOT / "checkpoints"


def _age_days(path: Path) -> float | None:
    if not path.is_file():
        return None
    sec = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
    return round(sec / 86400.0, 2)


def run_doctor(cfg: VeilConfig | None = None) -> dict[str, Any]:
    cfg = cfg or VeilConfig.load()
    report: dict[str, Any] = {
        "version": __version__,
        "python": sys.version.split()[0],
        "ok": True,
        "errors": [],
        "warnings": [],
    }
    try:
        import torch

        report["torch"] = getattr(torch, "__version__", "unknown")
        report["cuda"] = bool(torch.cuda.is_available())
        if report["cuda"]:
            report["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["torch"] = None
        report["cuda"] = False
        report["warnings"].append(f"torch: {type(exc).__name__}: {exc}")

    report["device"] = resolve_device(cfg.device)

    try:
        import jpeglib  # noqa: F401

        report["jpeglib"] = True
    except Exception:
        report["jpeglib"] = False
        report["warnings"].append("jpeglib missing; JSteg decode skipped")

    man_path = _ckpt_dir(cfg) / "manifest.json"
    expected: dict[str, str] = {}
    if man_path.is_file():
        import json

        man = json.loads(man_path.read_text(encoding="utf-8"))
        report["manifest"] = str(man_path)
        for row in man.get("models") or []:
            if row.get("name") and row.get("sha256"):
                expected[str(row["name"])] = str(row["sha256"]).lower()
    else:
        report["warnings"].append("checkpoints/manifest.json missing")

    ckpts = []
    for name, filename in (("residual_cnn", "residual_cnn.pt"), ("fsnet_lite", "fsnet_lite.pt")):
        path = _ckpt_dir(cfg) / filename
        row: dict[str, Any] = {"name": name, "path": str(path), "present": path.is_file()}
        if path.is_file():
            digest = sha256_file(path)
            row["sha256"] = digest
            row["bytes"] = path.stat().st_size
            want = expected.get(name)
            if want:
                row["expected"] = want
                row["ok"] = digest == want
                if digest != want:
                    report["ok"] = False
                    report["errors"].append(f"{name} sha256 drift")
            else:
                row["ok"] = True
                report["warnings"].append(f"{name} has no expected sha256 in manifest")
        else:
            row["ok"] = True
            row["skipped"] = True
            report["warnings"].append(f"{name} checkpoint missing (detector will skip)")
        ckpts.append(row)
    report["checkpoints"] = ckpts

    gen_op_path = Path(cfg.operating_point_path) if cfg.operating_point_path else ROOT / "configs" / "operating_point.json"
    gen = load_operating_point(gen_op_path if gen_op_path.is_file() else None)
    if not gen or str(gen.get("status") or "") not in {"provisional", "locked"}:
        report["ok"] = False
        report["errors"].append("generator operating_point missing or unset")
        report["operating_point"] = None
    else:
        report["operating_point"] = {
            "id": gen.get("id"),
            "status": gen.get("status"),
            "threshold": gen.get("threshold"),
            "path": str(gen_op_path),
            "age_days": _age_days(gen_op_path),
        }

    cam_path = ROOT / "configs" / "operating_point.camera.json"
    cam = load_operating_point(cam_path) if cam_path.is_file() else None
    if cam:
        report["camera_operating_point"] = {
            "id": cam.get("id"),
            "status": cam.get("status"),
            "threshold": cam.get("threshold"),
            "n": cam.get("n"),
            "path": str(cam_path),
        }
    else:
        report["camera_operating_point"] = None
        report["warnings"].append("camera operating point not measured")

    div_path = ROOT / "configs" / "operating_point.camera-div2k.json"
    div = load_operating_point(div_path) if div_path.is_file() else None
    if div:
        report["camera_div2k_operating_point"] = {
            "id": div.get("id"),
            "status": div.get("status"),
            "threshold": div.get("threshold"),
            "n": div.get("n"),
            "path": str(div_path),
        }
    else:
        report["camera_div2k_operating_point"] = None

    bench = ROOT / "docs" / "bench" / "latest.json"
    report["bench"] = {
        "path": str(bench),
        "present": bench.is_file(),
        "age_days": _age_days(bench),
    }
    return report
