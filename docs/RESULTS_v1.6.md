# VeilScan v1.6.0

Stage C confirmation. No recook of production FSNet. Generator lock
unchanged. BSDS 128px camera sidecar unchanged (`op-v1.4.0-camera-locked-n50`
~0.756). Default `present` stays ~0.67.

Package: `1.6.0`. Schema 5. `--policy` from v1.5.

## 1. DIV2K PNG pack (n=50, 128px)

`--corpus-id camera-div2k`. Files are PNG, so identity does **not** take
the JPEG-container blend. That is the point of this corpus.

| slice | family | TPR@5%FPR |
|-------|--------|-----------|
| identity | lsb | 0.980 |
| identity | dct | 0.780 |
| identity | dwt | 0.920 |
| jpeg_70 | dct | 0.600 |
| jpeg_70 | dwt | 0.860 |
| jpeg_50 | dct | 0.160 |
| jpeg_50 | dwt | 0.360 |
| jpeg_50 | tree_ring | 0.700 |

Cover mean identity 0.710, jpeg_70 0.571, jpeg_50 0.512.
Sidecar `op-v1.6.0-camera-div2k-locked-n50` ~0.796.
FPR at generator 0.67: **0.59**. FPR at BSDS sidecar 0.756: **0.12**.
Nested OR FPR 0.12 vs legacy 0.02. Default stays `legacy`.

Do not flip default `present`. PNG camera-like stills still sit on the
generator cut. JPEG-container blend is what made BSDS FPR 0.08.

## 2. BSDS n=50 at 256px (confirmation only)

Generator FPR at 0.67 is 0.02, but **TPR collapsed** (identity DCT 0.06,
DWT 0.04). Cover and marked means sit together. Do **not** replace the
128px BSDS lock. The 128px operating point remains canonical.

Likely cause: production FSNet and the eval generators were locked at
128px. Bigger stills are a different task, not a free upgrade.

## 3. LOAO holdout-dct

Fresh cook in `checkpoints/loao/` (gitignored): `--holdout-family dct`,
families dwt,spread,tree_ring, camera-train, 300 steps. Honest n=50 probe
on frozen BSDS test:

| family / attack | production TPR@5% | LOAO TPR@5% |
|-----------------|-------------------|-------------|
| dct / identity | 0.980 | **0.080** |
| dct / jpeg_70 | 0.860 | **0.060** |
| dct / jpeg_50 | 0.400 | **0.060** |
| dwt / identity | 1.000 | 1.000 |
| spread / identity | 1.000 | 1.000 |
| tree_ring / identity | 0.960 | 0.960 |

DCT AUC 0.998 -> 0.539 (chance). v1.2/v1.3 fine-tunes did **not** make
FSNet generic. Further cooks are still allowed under the existing rails.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67. File not overwritten.
- BSDS scan JSON `camera` sidecar stays v1.4 128px.
- Patchwork weight 0. No UniFreq. No remover. No from-scratch Q50.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
