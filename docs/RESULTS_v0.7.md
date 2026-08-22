# VeilScan v0.7.0

FSNet-lite recook on BSDS500 **train** stills mixed with synthetic photo
covers. Frozen BSDS500 **test-64** is still the camera bench. Generator lock
is unchanged. Default `present` still uses the generator cut. Fusion stays
`legacy`.

Package: `0.7.0`. Sidecar: `op-v0.7.0-camera-locked-n50`.
FSNet sha256: `6e45dbba4852c612679f3c1f957fc8040f0077c4826e0dfab9a9002b9472e580`.
ResidualCNN unchanged (`e82851366d533e86eaee9a48c75f6acc7be1166cd191641faa013b7cba37ac79`).

## Why this cut

v0.6 measured the hole: FSNet saturates on camera texture (`t_freq` 1.0,
cover mean ~1.0, frequency TPR@5%FPR near chance). Recooking on the frozen
test pack would leak the bench. Cook uses `data/covers/camera-train`
(BSDS500 train-200, same archive sha256 as test-64) with `--cover-mix 0.4`
and `--families dct,spread,dwt,tree_ring` (no lsb). `train_lite.py` refuses
`data/covers/camera` unless `--allow-test-covers`.

## Locked camera slice (n=50, size 128, BSDS500 test, identity + jpeg_70)

| slice | family | AUC | TPR@5%FPR |
|-------|--------|-----|-----------|
| identity | lsb | 1.000 | 1.000 |
| identity | dct | 0.993 | 0.960 |
| identity | dwt | 0.998 | 0.980 |
| identity | spread | 1.000 | 1.000 |
| identity | tree_ring | 0.985 | 0.900 |
| identity | patchwork | 0.648 | 0.020 |
| jpeg_70 | lsb | 0.528 | 0.080 |
| jpeg_70 | dct | 0.755 | 0.440 |
| jpeg_70 | dwt | 0.916 | 0.720 |
| jpeg_70 | spread | 0.856 | 0.540 |
| jpeg_70 | tree_ring | 0.992 | 0.940 |
| jpeg_70 | patchwork | 0.570 | 0.100 |

Cover mean identity ~0.709, jpeg_70 ~0.701. Camera ensemble threshold
**0.753** (FPR 0.05). `t_freq` **0.875** (was 1.0). Nested even/odd legacy
FPR 0.16, specialist-OR FPR 0.16. In-sample OR FPR 0.15 vs legacy 0.05.
Default stays legacy. Default `present` stays the generator lock.

v0.6 camera identity DCT TPR@5%FPR was 0.060. v0.7 is 0.960.

## Probe (FSNet only, n=32 camera test, not the ensemble)

| ckpt | camera cover mean | DCT id TPR@5% | DCT jpeg70 TPR@5% |
|------|-------------------|---------------|-------------------|
| v0.6 production | ~1.00 | 0.06 | 0.06 |
| camera-only cook | ~0.38 | 1.00 | 0.66 |
| v0.7 mix (shipped) | ~0.17 | 1.00 | 0.84 |

Camera-only cook recovered camera DCT but dropped generator-photo JPEG70 DCT
TPR@5% to 0.50. Mix 0.4 restored that slice (TPR 1.0 on n=16 synth photo)
and further cooled camera covers. LSB TPR@5% on FSNet stays 0 (ResidualCNN
is still the LSB specialist).

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` threshold ~0.67. See `docs/RESULTS_v0.5.md`.
Smoke n=20 photo identity+jpeg_70 after this cook: derived cut ~0.681.
Do not rewrite the locked file.

Generator lock ~0.67 is still below camera cover mean ~0.71, so it still
false-fires on these stills. Do not flip default `present` without a second
corpus.

## Still open

- Patchwork on camera stills stays unsupported (identity TPR@5%FPR 0.02).
- Camera JPEG70 DCT TPR 0.44 (identity 0.96). DWT 0.72 and tree_ring 0.94
  retain better.
- specialist-OR does not beat in-sample FPR. Do not flip.
- No UniFreq. No remover. No personal photos in git.
