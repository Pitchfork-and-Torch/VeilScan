# VeilScan v0.2.1 cook notes

CPU-only torch (`2.13.0+cpu`) on a machine that also has an RTX 4080. No CUDA wheel this turn.

## ResidualCNN

- Trained 50 steps, 96px, JPEG p=0.25, batch 8, AdamW 1e-3.
- `checkpoints/residual_cnn.pt` (~497 KB). `checkpoints/manifest.json` val_auc **0.739** on 48 val samples (high variance).
- Live 6-pair means for lsb/dct/dwt/spread were **flat** (cover=marked=0.511). Do not put this head in `peak_ok`. It exists so the detector **stops skipping**; treat scores as untrusted until a larger train or CUDA run.
- Resume training without a val early-stop *hurt* (val_auc 0.33). From-scratch 50 steps is the shipped file.
- ONNX: `checkpoints/residual_cnn.onnx` regenerable via `veilscan export-onnx`. Gitignored.

## Calibration

- Identity default: `configs/calibration.json` (tests stay stable).
- Fitted maps: `configs/calibration.fitted.json` (n=5, 80px, families lsb/dct/dwt/spread). Ensemble `a~2.24`, `b~-1.51`.
- To use fitted: copy over the identity file or point a future `calibration_path` at the fitted JSON. Do not enable fitted in default YAML without re-running pytest.

## Robustness (tiny)

DCT family, n=3, 80px, detectors `dct`+`chi_square`+`residual_cnn`:

| attack | AUC | retention |
|--------|-----|-----------|
| identity | 1.00 | 1.00 |
| jpeg_70 | 0.78 | 0.78 |
| noise | 1.00 | 1.00 |

Small-n. Not a photo FPR claim.

## Still not done

CUDA train, FSNet-lite weights, real DDIM inversion, UniFreq LOAO, TPR@1% FPR on photographs.
