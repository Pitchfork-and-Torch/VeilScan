"""Train ResidualCNN and FSNet-lite on synthetic pairs. CPU is slow; CUDA preferred."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from veilscan.config import resolve_device
from veilscan.eval.robustness import apply_attack
from veilscan.generators import FAMILIES, embed, synthetic_cover
from veilscan.models.fsnet import FSNetLite
from veilscan.models.residual_cnn import ResidualCNN

SIZE = 128
JPEG_CHOICES = ("jpeg_90", "jpeg_70", "jpeg_50")


class SyntheticWM(Dataset):
    def __init__(
        self,
        n: int,
        seed: int,
        holdout: str | None = None,
        jpeg_prob: float = 0.0,
        size: int = SIZE,
        cover_style: str = "mix",
    ) -> None:
        self.n = n
        self.size = size
        self.jpeg_prob = float(jpeg_prob)
        self.cover_style = cover_style
        self.rng = np.random.default_rng(seed)
        self.holdout = holdout
        self.families = [f for f in FAMILIES if f != holdout]
        if not self.families:
            raise ValueError("no training families left after holdout")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)) + idx)
        cover = synthetic_cover(self.size, self.size, rng, style=self.cover_style)
        if rng.random() < 0.5:
            fam = self.families[int(rng.integers(0, len(self.families)))]
            img = embed(cover, fam, seed=int(rng.integers(1 << 30)))
            y = 1.0
        else:
            img = cover
            y = 0.0
        if self.jpeg_prob > 0 and rng.random() < self.jpeg_prob:
            attack = JPEG_CHOICES[int(rng.integers(0, len(JPEG_CHOICES)))]
            img = apply_attack(img, attack, rng)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, torch.tensor(y, dtype=torch.float32)


def _val_auc(model: torch.nn.Module, loader: DataLoader, device: str, max_batches: int = 8) -> float:
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            logit = model(x.to(device))
            prob = torch.sigmoid(logit).cpu().numpy().reshape(-1)
            ps.append(prob)
            ys.append(y.numpy().reshape(-1))
    model.train()
    if not ys:
        return float("nan")
    y_a = np.concatenate(ys)
    s_a = np.concatenate(ps)
    pos = s_a[y_a >= 0.5]
    neg = s_a[y_a < 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = np.sum(pos[:, None] > neg[None, :])
    eq = np.sum(pos[:, None] == neg[None, :])
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def train_one(
    name: str,
    model: torch.nn.Module,
    steps: int,
    device: str,
    out: Path,
    holdout: str | None,
    jpeg_prob: float,
) -> dict:
    ds = SyntheticWM(max(steps * 4, 64), seed=0, holdout=holdout, jpeg_prob=jpeg_prob)
    val = SyntheticWM(64, seed=1, holdout=holdout, jpeg_prob=jpeg_prob)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    vloader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    n = 0
    last_loss = float("nan")
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
        last_loss = float(loss.detach().cpu())
        n += 1
        if n % 20 == 0:
            print(f"{name} step {n}/{steps} loss={last_loss:.4f}")
    auc = _val_auc(model, vloader, device)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"wrote {out} val_auc={auc:.3f}")
    return {
        "name": name,
        "steps": steps,
        "holdout_family": holdout,
        "jpeg_prob": jpeg_prob,
        "families": list(ds.families),
        "val_auc": auc,
        "last_loss": last_loss,
        "checkpoint": str(out.name),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "checkpoints"))
    p.add_argument("--holdout-family", default=None, help="Exclude this generator family from training (LOAO).")
    p.add_argument("--jpeg-prob", type=float, default=0.35, help="Probability of JPEG attack on each sample.")
    args = p.parse_args()
    device = resolve_device(args.device)
    print("device", device, "holdout", args.holdout_family, "jpeg_prob", args.jpeg_prob)
    root = Path(args.out)
    recs = []
    recs.append(
        train_one("residual_cnn", ResidualCNN(), args.steps, device, root / "residual_cnn.pt", args.holdout_family, args.jpeg_prob)
    )
    recs.append(
        train_one("fsnet_lite", FSNetLite(), args.steps, device, root / "fsnet_lite.pt", args.holdout_family, args.jpeg_prob)
    )
    manifest = {
        "schema_version": 1,
        "device": device,
        "holdout_family": args.holdout_family,
        "jpeg_prob": args.jpeg_prob,
        "steps": args.steps,
        "models": recs,
    }
    man_path = root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("wrote", man_path)


if __name__ == "__main__":
    main()
