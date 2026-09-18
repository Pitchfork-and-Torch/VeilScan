# VeilScan v1.7.0

Native 128 windows on large stills. No production recook. Generator lock
unchanged. BSDS 128px camera sidecar unchanged (`op-v1.4.0-camera-locked-n50`
~0.756). Default `present` stays ~0.67.

Package: `1.7.0`. Schema 5. `--policy` from v1.5.

## What 1.7 is

v1.6 measured BSDS at 256px and TPR collapsed (identity DCT 0.06). The
nets were trained at 128px. `deep.py` always area-resized the whole plate
to 128, which erases native 8x8 DCT structure on a 256 (or field-size)
raster.

v1.7 scores ResidualCNN and FSNet on native 128 windows: four corners plus
center, then the mean. Exact 128 plates stay one window and are not
resampled. Smaller plates still area-resize.

Stage D Q50 recooks from v1.3 weights were rejected first (see below).
This path does not touch `checkpoints/fsnet_lite.pt`.

## Locked 128px BSDS camera slice

Unchanged. Do not replace it with a 256px operating point. 128 n=16 smoke
after the window change still separates identity DCT (TPR 0.94). Default
`present` stays the generator lock.

## BSDS n=50 at 256px (confirmation, not a lock)

Same frozen test pack, size 256, attacks identity + jpeg_70 + jpeg_50.
v1.6 numbers in parentheses.

| slice | family | TPR@5%FPR | v1.6 |
|-------|--------|-----------|------|
| identity | lsb | 0.620 | 0.060 |
| identity | dct | 1.000 | 0.060 |
| identity | dwt | 1.000 | 0.040 |
| identity | spread | 1.000 | 0.060 |
| identity | tree_ring | 0.880 | 0.000 |
| jpeg_70 | dct | 0.900 | 0.080 |
| jpeg_70 | dwt | 0.940 | 0.040 |
| jpeg_70 | spread | 1.000 | 0.140 |
| jpeg_70 | tree_ring | 0.940 | 0.000 |
| jpeg_50 | dct | 0.400 | 0.060 |
| jpeg_50 | dwt | 0.620 | 0.100 |

Cover mean identity 0.534, jpeg_70 0.466, jpeg_50 0.458.
FPR at generator 0.67: **0.02**. FPR at BSDS sidecar 0.756: **0.00**.
Nested OR FPR 0.06 vs legacy 0.08 on the nested split; in-sample OR FPR
0.13 failed the 0.05 bar. Default stays `legacy`. Do not promote the
derived 256px threshold (~0.60).

## Stage D Q50 cook (rejected)

Two fine-tunes of production v1.3 weights, camera-train, no `lsb`, no
frozen test pack, no `--fresh`:

1. Heavier DCT/Q50, 300 steps, lr 1.5e-4. FSNet-head jpeg_50 DCT TPR
   0.40 -> **0.30**. Identity DCT 0.98 held.
2. Exact v1.3 recipe continuation, 300 steps, lr 2e-4. jpeg_50 DCT TPR
   0.40 -> **0.34**. Identity DCT 0.98 -> 1.00. AUC ticked 0.825 -> 0.839
   while TPR@5%FPR fell.

Production FSNet sha256 stays
`8064ca659e44bf7b832b3d847ce16a508a1b6d0c7692c90d69381f9d5b6bb195`.
JPEG50 DCT ensemble TPR at 128px remains **0.46**.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67. File not overwritten.
- BSDS scan JSON `camera` sidecar stays v1.4 128px.
- Patchwork weight 0. No UniFreq. No remover. No from-scratch Q50.
- Do not train FSNet with `lsb` or on frozen BSDS test.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
