# VeilScan v1.2.0

Q50-aware FSNet fine-tune. Presence ensemble plus keyless plaintext decode.
No remover. Generator lock unchanged. Native JPEG path from v1.1 stays.

Package: `1.2.0`. Scan JSON `schema_version` 4 (unchanged).
Camera sidecar: `op-v1.2.0-camera-locked-n50` ~0.484.
FSNet sha256: `2ea0ead9af468dbdc0bb1a1659a4275e2a75207f9e6f7129c9552f4a71faccc3`.
ResidualCNN unchanged.

## What 1.2 is

v1.1 made JPEG files take the blend. JPEG50 frequency TPR stayed weak
(DCT 0.18, DWT 0.42, spread 0.30) because the v1.0 cook saw Q50 on only
~23% of samples. v1.2 fine-tunes those weights on camera-train with a
Q50-heavy JPEG draw. A from-scratch Q50 cook was rejected: identity DCT
TPR collapsed 0.96 -> 0.38.

`--jpeg-attacks` on `train_lite.py` is the rail. Production cook:

```
py -3 scripts\train_lite.py --only fsnet_lite --steps 250 --lr 0.0003 --families dct,spread,dwt,tree_ring --jpeg-prob 0.75 --jpeg-attacks jpeg_50,jpeg_50,jpeg_70,jpeg_90 --covers data\covers\camera-train --cover-mix 0.35 --out checkpoints\candidates
```

Resumed v1.0 weights. No `lsb`. No frozen BSDS test pack.

## Locked BSDS camera slice (n=50, 128px)

Lock slice stays identity + jpeg_70. jpeg_50 is measured, not in the cut.

| slice | family | AUC | TPR@5%FPR | v1.1 TPR |
|-------|--------|-----|-----------|----------|
| identity | lsb | 0.916 | 0.100 | 0.060 |
| identity | dct | 0.996 | 0.960 | 1.000 |
| identity | dwt | 1.000 | 1.000 | 1.000 |
| identity | spread | 1.000 | 1.000 | 1.000 |
| identity | tree_ring | 0.992 | 0.940 | 0.940 |
| identity | patchwork | 0.613 | 0.080 | 0.060 |
| jpeg_70 | lsb | 0.503 | 0.040 | 0.060 |
| jpeg_70 | dct | 0.961 | 0.760 | 0.720 |
| jpeg_70 | dwt | 0.970 | 0.880 | 0.800 |
| jpeg_70 | spread | 0.988 | 0.900 | 0.860 |
| jpeg_70 | tree_ring | 0.991 | 0.980 | 0.900 |
| jpeg_70 | patchwork | 0.546 | 0.060 | 0.040 |
| jpeg_50 | lsb | 0.500 | 0.040 | 0.060 |
| jpeg_50 | dct | 0.733 | 0.320 | 0.180 |
| jpeg_50 | dwt | 0.856 | 0.560 | 0.420 |
| jpeg_50 | spread | 0.849 | 0.480 | 0.300 |
| jpeg_50 | tree_ring | 0.986 | 0.940 | 0.800 |
| jpeg_50 | patchwork | 0.505 | 0.060 | 0.060 |

Cover mean identity 0.364, jpeg_70 0.398, jpeg_50 0.427.
Combined lock threshold **0.484** (FPR 0.05). That sidecar is now
**below** the generator lock because unmarked JPEG scores fell. Default
`present` stays ~0.67. Nested OR FPR 0.16 vs legacy 0.08. Default stays
`legacy`.

At generator 0.67, BSDS camera FPR is **0.01** (v1.1 was 0.07, v1.0 was
0.66).

## Honest FSNet-head probe (n=50, both sides JPEG)

| family / attack | v1.0 TPR@5% | v1.2 TPR@5% |
|-----------------|-------------|-------------|
| dct / jpeg_50 | 0.26 | 0.26 |
| dwt / jpeg_50 | 0.44 | 0.54 |
| spread / jpeg_50 | 0.38 | 0.52 |
| tree_ring / jpeg_50 | 0.78 | 0.94 |
| dwt / jpeg_70 | 0.80 | 0.88 |
| dct / jpeg_70 | 0.76 | 0.82 |
| dct / identity | 0.98 | 0.98 |

The ensemble JPEG50 DCT lift (0.18 -> 0.32) is quieter covers plus the
v1.1 blend, not a smarter DCT head at Q50. DCT at Q50 remains the hole.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67. File not overwritten.
- DIV2K sidecar remains `op-v1.0.0-camera-div2k-locked-n50` (PNG pack).
- Patchwork unsupported (weight 0).
- No UniFreq. No remover. No personal photos in git.
- Do not train FSNet with `lsb` or on the frozen BSDS test pack.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
