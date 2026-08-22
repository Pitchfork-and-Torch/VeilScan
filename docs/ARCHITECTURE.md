# VeilScan architecture

## Layout

```
veilscan/
  src/veilscan/          importable package
    api.py               analyze() / analyze_path()
    types.py             DetectionResult, EnsembleResult
    registry.py          plugin discovery
    engine.py            tiered run + tiling
    ensemble.py          weighted fusion + uncertainty
    dsp.py               DCT / Haar / FFT / color / residuals
    image_io.py
    cli.py               typer entry
    config.py
    viz.py
    detectors/           one class per plugin, auto-registered
    generators/          eval-only embedders
    models/              FSNet-lite, residual CNN, WMD net, AE
    eval/                metrics, attacks, harness
  configs/default.yaml
  tests/
  examples/
  scripts/
  docs/
```

## Detector contract

Every detector subclasses `BaseDetector` and implements:

```
analyze(image: HxWxC uint8 RGB, context: AnalyzeContext) -> DetectionResult
```

`DetectionResult` fields:

- `detector`: stable slug
- `score`: [0, 1], higher => more likely watermarked
- `confidence`: [0, 1], how much to trust this score
- `heatmap`: optional HxW float
- `explanation`: one-paragraph human text
- `extras`: JSON-safe dict
- `skipped`: if true, ensemble weight is 0

`AnalyzeContext` carries config, device, optional clean reference images,
and whether deep checkpoints exist.

New detector: drop a module in `detectors/`, decorate with `@register`, done.

## Tiers

| Tier | Default members | When |
|------|-----------------|------|
| fast | chi-square, RS, SPA, bit-plane, histogram, patchwork, cheap DFT/DCT | always |
| frequency | DWT, hybrid, tree-ring spectral, Fourier-Mellin, block multi-scale | `--tier frequency` or `all` |
| residual | SRM-lite, color-space, reconstruction | `frequency` / `all` |
| deep | residual CNN, FSNet-lite | checkpoint on disk |
| blackbox | WMD | `--reference-dir` |
| foundation | patch-PCA / AE residual | `all` |

`veilscan scan` default tier is `all` but inactive modules skip themselves.

## Fusion

1. Drop skipped / zero-confidence results.
2. Mix score = `0.20*full_mean + 0.20*ok_mean + 0.20*top_mean + 0.40*peak`
   over `peak_ok` heads (chi-square, RS, bitplane, DCT, DWT, hybrid,
   tree-ring spectral, ResidualCNN, FSNet-lite). YAML weights still scale
   each head. Patchwork weight is 0 (unsupported on camera stills).
3. `jpeg_like` is true when the file is a JPEG container (`FF D8`) **or**
   8x8 blockiness >= 1.10. PNG / BMP / TIFF stay on pixel blockiness so
   a clean raster does not take the FSNet blend. When `jpeg_like` and
   FSNet ran, `score = 0.5 * mix + 0.5 * fsnet`. Scan JSON schema 4 adds
   `jpeg_container`, `jpeg_quality_est`, `jpeg_subsampling` on top of
   schema 3 `jpeg_like` / `jpeg_blockiness`.
4. `present = score >= threshold` from the generator operating point
   (`op-v0.4.0-locked-n50` ~0.67), not a hardcoded 0.55. Camera stills
   get a sidecar cut (~0.758) and do not replace default `present`.
5. Fusion mode stays `legacy`. `specialist_or` is measured, not default
   (nested OR FPR 0.20 vs legacy 0.12 on the v1.0 camera slice).
6. `uncertainty` = std of peak_ok scores. `confidence` =
   `clip(1 - uncertainty, 0, 1) * coverage`.
7. Combined heatmap: mean of resized per-detector maps.

## Image pipeline

Load (Pillow/OpenCV) -> RGB uint8 -> optional EXIF ignore (pixels only) ->
if `max(H,W) > tile_size`, overlapping tiles, per-tile detect, stitch heatmaps
by overlap-average, fuse tile scores by max (a mark in one tile is enough).

## Generators (eval only)

LSB, DCT mid-band, Haar-HH, SVD, Patchwork, spread-spectrum, tree-ring-approx,
hidden-approx (fixed high-pass kernel). Not a steganography product.

## Deep training

`scripts/train_lite.py` builds synthetic pairs from generators and trains
`ResidualCNN` / `FSNetLite` / WMD. Checkpoints land in `checkpoints/`.
Until those files exist, deep detectors skip.

## Evaluation

- Metrics: AUC, TPR at FPR 0.001 / 0.01 / 0.05, F1, PR-AUC, heatmap IoU if GT.
- Attacks: JPEG, resize, crop, noise, blur, jitter.
- Protocol helper: leave-one-family-out over generators.

Living numbers: `docs/RESULTS_v1.2.md` (v1.1: `docs/RESULTS_v1.1.md`).
Generator lock: `docs/RESULTS_v0.5.md`.
Historical maps: `docs/UPGRADE_PLAN.md`, `docs/NEXT_MASSIVE_UPGRADE.md`.

## Decode (v0.3+)

`veilscan.decode` is a separate walk from the presence ensemble. Container
comments first, then sequential LSB, then row scanlines and tile hotspots.
`veilscan inspect` runs scan then decode into one JSON and stamps the HUD.
JSON: `found`, `family` (`container` | `lsb` | `qr` | `jsteg` | `none`),
`text`, `layout`, `confidence`, optional `bbox`.

Eval planter: `veilscan embed-text` (not a hiding product).

## Non-goals

- Matching a closed SynthID verifier
- Diffusion inversion backend
- Watermark removal
- Mandatory GPU / DINOv2 download
- Universal payload decode (encrypted / neural / vendor)
