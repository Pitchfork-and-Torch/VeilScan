"""End-to-end demo: one clean cover vs LSB / DCT / tree-ring marked copies."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from veilscan.api import analyze
from veilscan.generators import embed, synthetic_cover
from veilscan.image_io import save_rgb
from veilscan.viz import save_overlay

OUT = Path(__file__).resolve().parent / "_demo_out"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    cover = synthetic_cover(192, 192, rng)
    save_rgb(OUT / "cover.png", cover)
    print("cover", analyze(cover).to_json()["score"])
    for fam in ("lsb", "dct", "tree_ring"):
        marked = embed(cover, fam, seed=11)
        save_rgb(OUT / f"{fam}.png", marked)
        result = analyze(marked)
        save_overlay(OUT / f"{fam}_heatmap.png", marked, result.heatmap)
        print(fam, result.to_json()["present"], round(result.score, 3), result.explanation)


if __name__ == "__main__":
    main()
