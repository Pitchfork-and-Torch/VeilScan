"""Train ResidualCNN and FSNet-lite on synthetic pairs. CPU is slow; CUDA preferred."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from veilscan.config import resolve_device
from veilscan.generators import FAMILIES, embed, synthetic_cover
from veilscan.models.fsnet import FSNetLite
from veilscan.models.residual_cnn import ResidualCNN

SIZE = 128


class SyntheticWM(Dataset):
    def __init__(self, n: int, seed: int) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.families = list(FAMILIES)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)) + idx)
        cover = synthetic_cover(SIZE, SIZE, rng)
        if rng.random() < 0.5:
            fam = self.families[int(rng.integers(0, len(self.families)))]
            img = embed(cover, fam, seed=int(rng.integers(1 << 30)))
            y = 1.0
        else:
            img = cover
            y = 0.0
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.tensor(y, dtype=torch.float32)


def train_one(name: str, model: torch.nn.Module, steps: int, device: str, out: Path) -> None:
    ds = SyntheticWM(max(steps * 4, 64), seed=0)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    n = 0
    it = iter(loader)
    while n < steps:
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logit = model(x)
        loss = loss_fn(logit, y)
        loss.backward()
        opt.step()
        n += 1
        if n % 20 == 0:
            print(f"{name} step {n}/{steps} loss={float(loss):.4f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "checkpoints"))
    args = p.parse_args()
    device = resolve_device(args.device)
    print("device", device)
    root = Path(args.out)
    train_one("residual_cnn", ResidualCNN(), args.steps, device, root / "residual_cnn.pt")
    train_one("fsnet_lite", FSNetLite(), args.steps, device, root / "fsnet_lite.pt")


if __name__ == "__main__":
    main()
