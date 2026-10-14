"""CLI wrapper. Prefer: py -3 -m veilscan export-onnx"""

from __future__ import annotations

import argparse
from pathlib import Path

from veilscan.export import export_residual_cnn


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="checkpoints/residual_cnn.onnx")
    p.add_argument("--ckpt", default="")
    p.add_argument("--size", type=int, default=128)
    args = p.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else None
    path = export_residual_cnn(Path(args.out), ckpt, args.size)
    print("wrote", path)


if __name__ == "__main__":
    main()
