"""ONNX export helpers. ResidualCNN only; FSNet FFT-DCT is not a reliable ONNX path."""

from __future__ import annotations

from pathlib import Path

import torch

from veilscan.models.residual_cnn import ResidualCNN


def export_residual_cnn(out: Path, checkpoint: Path | None = None, size: int = 128) -> Path:
    model = ResidualCNN()
    if checkpoint and checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()
    dummy = torch.zeros(1, 3, size, size)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["image"],
        output_names=["logit"],
        opset_version=17,
    )
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(out), **kwargs)
    return out
