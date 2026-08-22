# VeilScan v0.6.0

Camera stills operating point (BSDS500 test-64, n=50, 128px, identity + jpeg_70).
Generator lock is unchanged. Default `present` still uses the generator cut.
Scan JSON may add a `camera` sidecar. Fusion stays `legacy`.

Package: `0.6.0`. Sidecar: `op-v0.6.0-camera-locked-n50`.

## Why a second cut

Generator-photo covers sit near ensemble 0.64, so the 0.67 lock is FPR 0.05 there.
BSDS500 camera covers sit near ensemble **0.80**. The generator lock would call
most of those stills present. That is the measurement v0.5 asked for.

## Locked camera slice (n=50, size 128, BSDS500 test, identity + jpeg_70)

| slice | family | AUC | TPR@5%FPR |
|-------|--------|-----|-----------|
| identity | lsb | 1.000 | 1.000 |
| identity | dct | 0.649 | 0.060 |
| identity | dwt | 0.492 | 0.020 |
| identity | spread | 0.638 | 0.080 |
| identity | tree_ring | 0.668 | 0.120 |
| identity | patchwork | 0.618 | 0.080 |
| jpeg_70 | lsb | 0.520 | 0.080 |
| jpeg_70 | dct | 0.524 | 0.080 |
| jpeg_70 | dwt | 0.526 | 0.080 |
| jpeg_70 | spread | 0.595 | 0.100 |
| jpeg_70 | tree_ring | 0.749 | 0.320 |
| jpeg_70 | patchwork | 0.546 | 0.080 |

Cover mean identity ~0.796, jpeg_70 ~0.784. Camera ensemble threshold **0.817**
(FPR 0.05 on that combined slice). `fpr_est` 0.05. Nested even/odd legacy FPR
0.06, specialist-OR FPR 0.98. Default stays legacy. Default `present` stays
the generator lock.

In-sample A/B at the camera cut: legacy LSB TPR 0.52, DCT TPR 0.11 (jpeg_70
half of the slice is mostly dead). OR FPR 0.92. Do not flip.

## What the camera numbers mean

- **LSB identity still separates** on these stills (AUC 1.0 at 5% FPR).
- **JPEG70 kills spatial LSB** (AUC 0.52), same lesson as the generator slice.
- **FSNet saturates on camera texture.** Frequency-head mean cover is ~1.0, so
  `t_freq` pins at 1.0. The net cannot cut 5% FPR on this pack. Do not read
  generator JPEG70 DCT TPR (0.94) as camera DCT TPR.
- Frequency families barely move the ensemble (cover 0.796 vs marked ~0.80).
  That is a camera-texture problem, not a reason to raise YAML weights.

Do not cite this as UniFreq or ImageNet FPR. Extracted JPEGs stay gitignored.
Archive sha256 is in `configs/camera_covers.manifest.json`.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` threshold ~0.67. See `docs/RESULTS_v0.5.md`.
