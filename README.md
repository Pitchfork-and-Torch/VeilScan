# VeilScan

Local, algorithm-agnostic **invisible watermark presence detector** with a **keyless plaintext** reader.

It answers: *does this image contain an invisible watermark?* and, when the payload is sequential LSB / PNG text / JPEG comment, *what does the text say?* It does not strip a mark. It does not break SynthID, Digimarc, encrypted stego, or neural watermarks.

v0.3 ships a plugin ensemble of classical spatial tests, multi-transform frequency analysis, residual / SRM-lite features, optional FSNet-lite and residual CNNs (skip until checkpoints exist), WMD-style offset learning when you have a clean reference set, and `veilscan decode` for keyless plaintext.

## Synthetic selftest (v0, n=4 covers/family, 128px)

`py -3 -m veilscan selftest`

| Family | AUC (rank) | Notes |
|--------|------------|-------|
| LSB | ~0.94 | Chi-square / RS; thresholded F1 lags rank |
| DCT / DWT / spread / patchwork | ~1.0 on this draw | Strongest classical-frequency hits |
| Tree-Ring approx | ~0.8 rank, weak F1 | Pixel FFT proxy, not DDIM inversion |
| SVD / hidden-approx | weak / unstable | Honest miss; needs a trained residual net |

Overall on that draw: AUC ~0.83, TPR@5% FPR ~0.59. Small-n selftest variance is real. This is **not** UniFreq-100K leave-one-algorithm-out. Deep plugins are off until you train checkpoints.

## Install

```powershell
cd $env:USERPROFILE\veilscan
py -3 -m pip install -e .
py -3 -m pytest
```

Optional extras: `pip install -e ".[dev,gui]"`

## Quick start

```powershell
py -3 -m veilscan list-detectors
py -3 -m veilscan scan path\to\image.png
py -3 -m veilscan scan path\to\image.png --json --heatmap out_overlay.png
py -3 -m veilscan batch path\to\folder --json
py -3 -m veilscan decode path\to\image.png
# writes path-veilscan-report.png (HUD overlay + executive brief) next to the file
py -3 -m veilscan selftest
py -3 -m veilscan selftest --per-detector
py -3 -m veilscan loao --n 3
```

Python:

```python
from veilscan import analyze_path, decode_path
r = analyze_path("path/to/image.png")
print(r.present, r.score, r.explanation)
d = decode_path("path/to/image.png")
print(d.found, d.family, d.text)
```

## What it implements

| Pipeline | Plugins |
|----------|---------|
| Spatial | chi-square, RS, sample pairs, bit-plane entropy, histogram comb, patchwork-style pairs |
| Frequency | block DCT, DFT 1/f residual, Haar DWT, DWT-DCT-SVD hybrid, Tree-Ring circular FFT, Fourier-Mellin, multi-scale blocks |
| Residual | SRM-lite KV + co-occurrence, YCbCr/HSV, denoising reconstruction, higher-order / bispectrum proxy |
| Deep | ResidualCNN, FSNet-lite (ASPM + DMSA). **Skipped if `checkpoints/*.pt` missing.** |
| Black-box | WMD offset learning. **Skipped without `--reference-dir`.** |
| Foundation | patch-PCA residual (no mandatory DINOv2 download) |

Research notes: `docs/RESEARCH.md`. Architecture: `docs/ARCHITECTURE.md`. Limits: `docs/LIMITATIONS.md`. Ethics: `docs/ETHICS.md`.

## Train the tiny nets (optional)

```powershell
py -3 scripts\train_lite.py --steps 80
```

Needs PyTorch. CUDA is faster; CPU works. Until checkpoints exist, the ensemble is classical + frequency only.

## Eval watermarks

`veilscan embed`, `veilscan embed-text`, and `veilscan.generators` exist **only** to test the detector and decoder. This is not a steganography product. There is no remover.

## GUI / API

`py -3 -m veilscan.gui` starts Gradio. Do **not** launch it from a Grok Build command (never-exit servers hang the TUI Job Object). FastAPI extra is declared but not the default path.

## Honest limits

Leave-one-algorithm-out work (AWPD / FSNet, UniFreq-100K) shows LSB and Patchwork defeat frequency-centric nets. Latent-only marks (Gaussian Shading, some Tree-Ring / SynthID cases) are weak in pixel space without inversion or a vendor verifier. A VeilScan score is not a copyright ruling.

## Docs

- `docs/RESEARCH.md` -- AWPD survey
- `docs/ARCHITECTURE.md` -- plugin contract
- `docs/LIMITATIONS.md` / `docs/ETHICS.md`
- `docs/UPGRADE_PLAN.md` -- v0.1.0 -> v0.2 map (historical)
- `docs/NEXT_MASSIVE_UPGRADE.md` -- v0.4 operating-point contract (bench). v0.3.0 is keyless decode.
- `docs/PHASES_2_3_4.md` -- calibration, WMD prune, batch/ONNX (v0.2.0)
- `docs/RESULTS_v0.2.1.md` -- CPU ResidualCNN cook (flat scores; superseded)
- `docs/RESULTS_v0.2.2.md` -- CUDA ResidualCNN with LSB planes; peak_ok
- `docs/RESULTS_v0.2.3.md` -- FSNet LSB stem; both deep heads in peak_ok
- `docs/RESULTS_v0.2.4.md` -- FSNet frequency cook (no LSB family); JPEG70 DCT holds

`py -3 -m veilscan loao` and `selftest --per-detector` are the Phase 0 measurement CLI.

## License

Proprietary. All rights reserved. Independent reimplementation of published ideas (RS, SRM, FSNet, WMD). Not the authors' official code. Not a public product.
