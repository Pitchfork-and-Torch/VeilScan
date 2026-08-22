"""Score one or two FSNet checkpoints on camera covers vs a marked family.

Does not write production checkpoints. Does not hit the network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from veilscan.eval.bench import load_camera_covers
from veilscan.eval.metrics import auc_roc, tpr_at_fpr
from veilscan.eval.robustness import apply_attack
from veilscan.generators import embed
from veilscan.models.fsnet import FSNetLite

ROOT = Path(__file__).resolve().parents[1]


def _load(ckpt: Path, device: str) -> torch.nn.Module:
    model = FSNetLite()
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _score(model: torch.nn.Module, rgb: np.ndarray, device: str) -> float:
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logit = model(x.to(device))
        if logit.ndim > 0:
            logit = logit.reshape(-1)[0]
        return float(torch.sigmoid(logit).cpu())


def probe_ckpt(
    model: torch.nn.Module,
    covers: list[np.ndarray],
    *,
    family: str,
    attack: str,
    device: str,
    seed: int,
) -> dict:
    y: list[int] = []
    s: list[float] = []
    rng = np.random.default_rng(seed)
    for i, cover in enumerate(covers):
        marked = embed(cover, family, seed=seed + i)
        if attack != "identity":
            marked = apply_attack(marked, attack, rng)
        s.append(_score(model, cover, device))
        y.append(0)
        s.append(_score(model, marked, device))
        y.append(1)
    y_a = np.asarray(y)
    s_a = np.asarray(s, dtype=np.float64)
    return {
        "n_cover": int((y_a == 0).sum()),
        "mean_cover": float(s_a[y_a == 0].mean()),
        "mean_marked": float(s_a[y_a == 1].mean()),
        "auc": auc_roc(y_a, s_a),
        "tpr_at_fpr_5": tpr_at_fpr(y_a, s_a, 0.05),
        "family": family,
        "attack": attack,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Probe FSNet on covers vs one embed family.")
    p.add_argument("--covers", required=True, help="Folder of stills (use frozen test pack to compare cooks).")
    p.add_argument("--ckpt-a", default=str(ROOT / "checkpoints" / "fsnet_lite.pt"))
    p.add_argument("--ckpt-b", default="", help="Optional candidate checkpoint.")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--family", default="dct")
    p.add_argument("--attack", default="identity", help="identity or jpeg_70")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    covers = load_camera_covers(args.covers, args.n, args.size, seed=20260822)
    report: dict = {
        "n": len(covers),
        "family": args.family,
        "attack": args.attack,
        "device": device,
        "covers": str(Path(args.covers)),
    }
    for key, path_s in (("a", args.ckpt_a), ("b", args.ckpt_b)):
        if not path_s:
            continue
        path = Path(path_s)
        if not path.is_file():
            print(f"FAIL missing {path}", flush=True)
            return 2
        model = _load(path, device)
        stats = probe_ckpt(model, covers, family=args.family, attack=args.attack, device=device, seed=20260822)
        stats["ckpt"] = str(path)
        report[key] = stats
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
