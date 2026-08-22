# VeilScan

Local, algorithm-agnostic **invisible watermark presence detector** with a **keyless plaintext** reader.

It answers: *does this image contain an invisible watermark?* and, when the payload is sequential LSB / PNG text / JPEG comment, *what does the text say?* It does not strip a mark. It does not break SynthID, Digimarc, encrypted stego, or neural watermarks.

Lamp: https://veilscan.jonbailey.xyz/  
Private GitHub: https://github.com/Pitchfork-and-Torch/VeilScan

v1.2.0 fine-tunes FSNet for JPEG Q50 without a from-scratch recook. Native JPEG path from v1.1 stays: file bytes set `jpeg_container`. Generator `present` stays ~0.67. On BSDS camera JPEGs that cut false-fires at 0.01. Camera sidecar `op-v1.2.0-camera-locked-n50` ~0.484. JPEG50 DWT TPR 0.42 -> 0.56. JPEG50 DCT is still weak (0.32). Patchwork remains unsupported (weight 0). ResidualCNN unchanged. Do not vendor UniFreq or personal photos in git.

## Synthetic numbers

`py -3 -m veilscan selftest` is a fast n=6 @ 128px smoke, not an operating point.

`py -3 -m veilscan bench --n 50 --styles photo --attacks identity,jpeg_70 --write-operating-point` writes `docs/bench/latest.json` from `configs/bench_protocol.yaml`. That corpus is **generator-photo**, not ImageNet, not UniFreq-100K. Locked `op-v0.4.0-locked-n50`: ensemble threshold ~0.67 (FPR 0.05 on that slice). Camera sidecar `op-v1.2.0-camera-locked-n50`: ~0.484 (FPR 0.05 on BSDS500 test-64). DIV2K confirmation `op-v1.0.0-camera-div2k-locked-n50`: ~0.777 (PNG pack, not re-locked). JPEG files and JPEG-like arrays blend 0.5 FSNet into the legacy mix. `--corpus-id` keeps a second pack from clobbering the BSDS sidecar. `veilscan inspect` runs scan then decode into one JSON/HUD. Numbers: `docs/RESULTS_v1.2.md`.

Complementary split (do not mix):

- ResidualCNN trains on LSB-plane cues. Do not strip bit planes.
- FSNet-lite must **not** see `--families lsb` (the cook exits 2 unless `--force`).

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
py -3 -m veilscan doctor
py -3 -m veilscan inspect path\to\image.png
py -3 -m veilscan selftest
py -3 -m veilscan selftest --per-detector
py -3 -m veilscan bench --n 50 --styles photo --attacks identity,jpeg_70 --write-operating-point
py -3 scripts\fetch_camera_covers.py --dry-run
py -3 scripts\fetch_camera_covers.py
py -3 scripts\fetch_camera_covers.py --manifest configs\camera_train_covers.manifest.json --out data\covers\camera-train
py -3 scripts\train_lite.py --only fsnet_lite --steps 250 --lr 0.0003 --families dct,spread,dwt,tree_ring --jpeg-prob 0.75 --jpeg-attacks jpeg_50,jpeg_50,jpeg_70,jpeg_90 --covers data\covers\camera-train --cover-mix 0.35 --out checkpoints\candidates
py -3 scripts\probe_fsnet.py --covers data\covers\camera --ckpt-a checkpoints\fsnet_lite.pt --ckpt-b checkpoints\candidates\fsnet_lite.pt
py -3 -m veilscan bench --covers data\covers\camera --n 50 --attacks identity,jpeg_70 --write-operating-point
py -3 scripts\fetch_camera_covers.py --manifest configs\camera_div2k_covers.manifest.json --out data\covers\camera-div2k
py -3 -m veilscan bench --covers data\covers\camera-div2k --corpus-id camera-div2k --n 50 --attacks identity,jpeg_70 --write-operating-point
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
py -3 scripts\train_lite.py --only residual_cnn --steps 80
py -3 scripts\train_lite.py --only fsnet_lite --families dct,dwt,spread,tree_ring --steps 80
```

Needs PyTorch. CUDA is faster; CPU works. Checkpoints already live in this tree. Passing `lsb` to the FSNet cook exits 2.

## Eval watermarks

`veilscan embed`, `veilscan embed-text`, and `veilscan.generators` exist **only** to test the detector and decoder. This is not a steganography product. There is no remover.

## GUI / API

`py -3 -m veilscan.gui` starts Gradio. Do **not** launch it from a Grok Build command (never-exit servers hang the TUI Job Object). FastAPI extra is declared but not the default path.

## Honest limits

Leave-one-algorithm-out work (AWPD / FSNet, UniFreq-100K) shows LSB and Patchwork defeat frequency-centric nets. Latent-only marks (Gaussian Shading, some Tree-Ring / SynthID cases) are weak in pixel space without inversion or a vendor verifier. A VeilScan score is not a copyright ruling.

## Docs

- `docs/RESULTS_v1.2.md` -- current numbers (Q50-aware FSNet, camera OP)
- `docs/RESULTS_v1.1.md` -- native JPEG path
- `docs/RESULTS_v1.0.md` -- JPEG-hardened FSNet cook and LOAO
- `docs/RESEARCH.md` -- AWPD survey
- `docs/ARCHITECTURE.md` -- plugin contract and fusion
- `docs/LIMITATIONS.md` / `docs/ETHICS.md`
- `docs/RESULTS_v0.5.md` -- generator lock `op-v0.4.0-locked-n50`
- `docs/UPGRADE_PLAN.md` -- v0.1.0 -> v0.2 map (historical)
- `docs/NEXT_MASSIVE_UPGRADE.md` -- v0.4 operating-point contract (shipped)
- `docs/PHASES_2_3_4.md` -- calibration, WMD prune, batch/ONNX (v0.2.0)
- `docs/RESULTS_v0.2.1.md` through `RESULTS_v0.9.md` -- cook notes (superseded)

`py -3 -m veilscan loao` and `selftest --per-detector` are the Phase 0 measurement CLI.

## License

Proprietary. All rights reserved. Independent reimplementation of published ideas (RS, SRM, FSNet, WMD). Not the authors' official code. Not a public product.
