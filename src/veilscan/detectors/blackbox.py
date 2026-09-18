"""WMD-style offset learning. Dataset-level; skipped without a clean reference set."""

from __future__ import annotations

import numpy as np

from veilscan.detectors.base import BaseDetector
from veilscan.registry import register
from veilscan.types import AnalyzeContext, DetectionResult


class WMDDetector(BaseDetector):
    name = "wmd"
    tier = "blackbox"
    requires_reference = True

    def analyze(self, image: np.ndarray, context: AnalyzeContext | None = None) -> DetectionResult:
        refs = context.reference_images if context else None
        if not refs:
            return self.skip("WMD needs --reference-dir (clean images of similar distribution)")
        # Single-image mode: one gradient-offset step against the reference batch.
        try:
            score, extras = _wmd_one(image, refs, device=(context.device if context else "cpu"))
        except Exception as e:
            return self.skip(f"WMD failed: {e}")
        expl = (
            f"WMD offset-learning (Pan et al. ECCV 2024 idea). score={score:.3f}. "
            "Uses a clean reference set; not a decoder."
        )
        return DetectionResult(self.name, score, 0.55, expl, extras=extras, tier=self.tier).clamp()


def _wmd_one(image: np.ndarray, refs: list[np.ndarray], device: str, steps: int = 8) -> tuple[float, dict]:
    import cv2
    import torch

    from veilscan.models.wmd_net import WMDNet

    def prep(im: np.ndarray) -> torch.Tensor:
        im = cv2.resize(im, (96, 96), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t

    model = WMDNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    suspect = prep(image).unsqueeze(0).to(device)
    clean = torch.stack([prep(r) for r in refs[:16]], dim=0).to(device)
    tau = 0.5
    for _ in range(steps):
        opt.zero_grad()
        pc = model(clean)
        pd = model(suspect.expand(clean.size(0), -1, -1, -1))
        # Asymmetric: softmax-ish on clean (minimize), linear on suspect (maximize).
        loss_clean = (tau * torch.logsumexp(pc / tau, dim=0))
        loss_det = -pd.mean()
        loss = loss_clean + loss_det
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = float(model(suspect).mean().cpu())
        c = float(model(clean).mean().cpu())
    # Higher suspect output than clean mean => watermark-like under offset learning.
    gap = s - c
    score = float(1.0 / (1.0 + np.exp(-gap)))
    return score, {"suspect_logit": s, "clean_logit": c, "gap": gap, "n_ref": int(clean.size(0))}


def wmd_prune_scan(
    suspects: list[np.ndarray],
    refs: list[np.ndarray],
    *,
    rounds: int = 2,
    steps: int = 4,
    keep: float = 0.5,
    device: str = "cpu",
) -> list[dict]:
    """Dataset-level WMD: train, score, drop lowest, repeat. Detection only."""
    if not suspects or not refs:
        raise ValueError("WMD prune needs suspects and clean references")
    import cv2
    import torch

    from veilscan.models.wmd_net import WMDNet

    def prep(im: np.ndarray) -> torch.Tensor:
        im = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(im.astype(np.float32) / 255.0).permute(2, 0, 1)

    clean = torch.stack([prep(r) for r in refs[:16]], dim=0).to(device)
    active = list(range(len(suspects)))
    scores = np.full(len(suspects), 0.5, dtype=np.float64)
    history = []
    tau = 0.5
    keep = float(np.clip(keep, 0.2, 0.9))
    for rnd in range(max(1, rounds)):
        model = WMDNet().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = torch.stack([prep(suspects[i]) for i in active], dim=0).to(device)
        for _ in range(max(1, steps)):
            opt.zero_grad()
            pc = model(clean)
            pd = model(batch)
            loss = tau * torch.logsumexp(pc / tau, dim=0) - pd.mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = model(batch).detach().cpu().numpy().reshape(-1)
            cmean = float(model(clean).mean().cpu())
        for i, idx in enumerate(active):
            gap = float(logits[i] - cmean)
            scores[idx] = 1.0 / (1.0 + np.exp(-gap))
        history.append({"round": rnd, "n_active": len(active), "mean_score": float(scores[active].mean())})
        if len(active) <= 2:
            break
        order = sorted(active, key=lambda i: scores[i], reverse=True)
        n_keep = max(2, int(np.ceil(len(active) * keep)))
        active = order[:n_keep]
    return {
        "items": [
            {"index": i, "score": float(scores[i]), "survived": i in set(active)}
            for i in range(len(suspects))
        ],
        "history": history,
        "n_suspects": len(suspects),
        "n_ref": int(clean.size(0)),
    }


def register_blackbox() -> None:
    register(WMDDetector())
