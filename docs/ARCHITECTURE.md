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
2. `score = sum(w_i * s_i) / sum(w_i)` with YAML weights.
3. `uncertainty = population std of active scores` (disagreement).
4. `confidence = clip(1 - uncertainty, 0, 1) * coverage` where coverage is
   fraction of expected detectors that actually ran.
5. `present = score >= threshold` (default 0.55).
6. Combined heatmap: mean of resized per-detector maps.

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

Living numbers: `docs/RESULTS_v0.5.md`. Historical maps: `docs/UPGRADE_PLAN.md`,
`docs/NEXT_MASSIVE_UPGRADE.md`.

## Decode (v0.3+)

`veilscan.decode` is a separate walk from the presence ensemble. Container
comments first, then sequential LSB, then row scanlines and tile hotspots.
`veilscan inspect` runs scan then decode into one JSON and stamps the HUD.
JSON: `found`, `family` (`container` | `lsb` | `qr` | `jsteg` | `none`),
`text`, `layout`, `confidence`, optional `bbox`.

Eval planter: `veilscan embed-text` (not a hiding product).

## Non-goals (v0)

- Matching a closed SynthID verifier
- Diffusion inversion backend
- Watermark removal
- Mandatory GPU / DINOv2 download
- Universal payload decode (encrypted / neural / vendor)
