"""Train ResidualCNN and FSNet-lite on synthetic pairs. CPU is slow; CUDA preferred."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from veilscan.config import resolve_device
from veilscan.eval.bench import iter_cover_paths
from veilscan.eval.robustness import apply_attack
from veilscan.generators import FAMILIES, embed, synthetic_cover
from veilscan.image_io import load_rgb
from veilscan.models.fsnet import FSNetLite
from veilscan.models.residual_cnn import ResidualCNN

SIZE = 128
JPEG_CHOICES = ("jpeg_90", "jpeg_70", "jpeg_50")
ROOT = Path(__file__).resolve().parents[1]
FROZEN_TEST_COVERS = ROOT / "data" / "covers" / "camera"
DEFAULT_CKPT = ROOT / "checkpoints"


def resize_cover(rgb: np.ndarray, size: int) -> np.ndarray:
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("RGB cover required")
    return cv2.resize(rgb[..., :3], (size, size), interpolation=cv2.INTER_AREA)


def split_cover_paths(paths: list[Path], seed: int = 0, val_frac: float = 0.2) -> tuple[list[Path], list[Path]]:
    if not paths:
        raise ValueError("no cover paths")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_val = max(1, int(round(len(paths) * val_frac)))
    if n_val >= len(paths):
        n_val = max(1, len(paths) // 5) if len(paths) >= 5 else 1
        n_val = min(n_val, len(paths) - 1) if len(paths) > 1 else 1
    val_idx = set(int(i) for i in order[:n_val].tolist())
    train = [paths[int(i)] for i in order if int(i) not in val_idx]
    val = [paths[int(i)] for i in order if int(i) in val_idx]
    if not train:
        train = list(val)
    return train, val


def assert_not_frozen_test(cover_dir: Path, *, allow: bool) -> None:
    if allow:
        return
    try:
        if cover_dir.resolve() == FROZEN_TEST_COVERS.resolve():
            raise SystemExit(
                "ERROR: refuse to train on the frozen BSDS500 test pack "
                f"({FROZEN_TEST_COVERS}). Fetch train stills with "
                "scripts/fetch_camera_covers.py --manifest configs/camera_train_covers.manifest.json "
                "--out data/covers/camera-train. Override only with --allow-test-covers."
            )
    except FileNotFoundError:
        return


class SyntheticWM(Dataset):
    def __init__(
        self,
        n: int,
        seed: int,
        holdout: str | None = None,
        jpeg_prob: float = 0.0,
        size: int = SIZE,
        cover_style: str = "mix",
        families: list[str] | None = None,
        cover_paths: list[Path] | None = None,
        cover_mix: float = 0.0,
    ) -> None:
        self.n = n
        self.size = size
        self.jpeg_prob = float(jpeg_prob)
        self.cover_style = cover_style
        self.rng = np.random.default_rng(seed)
        self.holdout = holdout
        self.cover_paths = list(cover_paths or [])
        self.cover_mix = float(cover_mix)
        src = list(families) if families else list(FAMILIES)
        self.families = [f for f in src if f != holdout]
        if not self.families:
            raise ValueError("no training families left after holdout")

    def __len__(self) -> int:
        return self.n

    def _cover(self, rng: np.random.Generator, idx: int) -> np.ndarray:
        use_disk = bool(self.cover_paths) and rng.random() >= self.cover_mix
        if use_disk:
            path = self.cover_paths[idx % len(self.cover_paths)]
            return resize_cover(load_rgb(path), self.size)
        return synthetic_cover(self.size, self.size, rng, style=self.cover_style)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)) + idx)
        cover = self._cover(rng, idx)
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
    size: int = SIZE,
    fresh: bool = False,
    families: list[str] | None = None,
    batch_size: int = 8,
    lr: float = 1e-3,
    cover_style: str = "mix",
    cover_paths: list[Path] | None = None,
    val_paths: list[Path] | None = None,
    cover_mix: float = 0.0,
) -> dict:
    if fresh and out.is_file():
        out.unlink()
    ds = SyntheticWM(
        max(steps * 8, 128),
        seed=0,
        holdout=holdout,
        jpeg_prob=jpeg_prob,
        size=size,
        families=families,
        cover_style=cover_style,
        cover_paths=cover_paths,
        cover_mix=cover_mix,
    )
    val = SyntheticWM(
        160,
        seed=1,
        holdout=holdout,
        jpeg_prob=jpeg_prob,
        size=size,
        families=families,
        cover_style=cover_style,
        cover_paths=val_paths if val_paths is not None else cover_paths,
        cover_mix=cover_mix,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    vloader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(device)
    if out.is_file():
        state = torch.load(out, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"{name} resumed {out}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
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
    auc = _val_auc(model, vloader, device, max_batches=20)
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
        "n_cover_paths": len(cover_paths or []),
        "n_val_paths": len(val_paths or []),
        "cover_mix": cover_mix,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(DEFAULT_CKPT))
    p.add_argument("--holdout-family", default=None, help="Exclude this generator family from training (LOAO).")
    p.add_argument("--jpeg-prob", type=float, default=0.35, help="Probability of JPEG attack on each sample.")
    p.add_argument("--size", type=int, default=SIZE)
    p.add_argument("--only", default="both", help="residual_cnn | fsnet_lite | both")
    p.add_argument("--fresh", action="store_true", help="Do not resume an existing checkpoint.")
    p.add_argument("--families", default="", help="Comma list; default all generators.")
    p.add_argument("--batch-size", type=int, default=0, help="0 = 32 on cuda else 8")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cover-style", default="mix", help="sine | photo | mix")
    p.add_argument("--covers", default="", help="Folder of camera stills. Never the frozen test pack.")
    p.add_argument("--cover-mix", type=float, default=0.0, help="Prob of synthetic cover when --covers is set.")
    p.add_argument("--allow-test-covers", action="store_true", help="Allow training on data/covers/camera (leaks the bench).")
    p.add_argument("--force", action="store_true", help="Allow FSNet to see lsb (destroys the frequency cue).")
    args = p.parse_args()
    device = resolve_device(args.device)
    fams = [x.strip() for x in args.families.split(",") if x.strip()] or None
    train_paths: list[Path] | None = None
    val_paths: list[Path] | None = None
    if args.covers:
        cover_dir = Path(args.covers)
        assert_not_frozen_test(cover_dir, allow=args.allow_test_covers)
        paths = iter_cover_paths(cover_dir)
        if not paths:
            raise SystemExit(f"ERROR: no images under {cover_dir}")
        train_paths, val_paths = split_cover_paths(paths, seed=0, val_frac=0.2)
        print("covers", cover_dir, "train", len(train_paths), "val", len(val_paths), "mix", args.cover_mix)
    if args.only in ("both", "fsnet_lite"):
        used = list(fams) if fams else list(FAMILIES)
        if "lsb" in used:
            print(
                "ERROR: FSNet-lite must not train on lsb. The frequency cue collapses. "
                "Cook ResidualCNN separately for LSB, and pass --families without lsb for FSNet. "
                "Override only with --force."
            )
            if not args.force:
                raise SystemExit(2)
    bs = args.batch_size or (32 if device == "cuda" else 8)
    print("device", device, "holdout", args.holdout_family, "jpeg_prob", args.jpeg_prob, "only", args.only, "bs", bs, "families", fams)
    root = Path(args.out)
    if root.resolve() == DEFAULT_CKPT.resolve():
        print("WARN: --out is production checkpoints/. Prefer checkpoints/candidates until a probe beats the current FSNet.")
    recs = []
    if args.only in ("both", "residual_cnn"):
        recs.append(
            train_one(
                "residual_cnn",
                ResidualCNN(),
                args.steps,
                device,
                root / "residual_cnn.pt",
                args.holdout_family,
                args.jpeg_prob,
                size=args.size,
                fresh=args.fresh,
                families=fams,
                batch_size=bs,
                lr=args.lr,
                cover_style=args.cover_style,
                cover_paths=train_paths,
                val_paths=val_paths,
                cover_mix=args.cover_mix,
            )
        )
    if args.only in ("both", "fsnet_lite"):
        recs.append(
            train_one(
                "fsnet_lite",
                FSNetLite(),
                args.steps,
                device,
                root / "fsnet_lite.pt",
                args.holdout_family,
                args.jpeg_prob,
                size=args.size,
                fresh=args.fresh,
                families=fams,
                batch_size=bs,
                lr=args.lr,
                cover_style=args.cover_style,
                cover_paths=train_paths,
                val_paths=val_paths,
                cover_mix=args.cover_mix,
            )
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
