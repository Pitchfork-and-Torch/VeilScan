# VeilScan v1.4.0

Quality-aware FSNet blend. Schema 5. No recook. Generator lock unchanged.

Package: `1.4.0`. Scan JSON `schema_version` 5 adds `jpeg_freq_weight`.
Camera sidecar: `op-v1.4.0-camera-locked-n50` ~0.756.
FSNet and ResidualCNN weights unchanged from v1.3.

## What 1.4 is

v1.3 used a binary `jpeg_like` switch at blend 0.5. Q95 camera JPEGs and
Q50 re-encodes got the same mix. v1.4 sets `jpeg_freq_weight` from
`jpeg_quality_est` (blockiness fallback if quality is missing):

- not JPEG: 0
- Q >= 90: 0.25
- Q ~ 70: ~0.48
- Q <= 50: 0.70

## Locked BSDS camera slice (n=50, 128px)

| slice | family | TPR@5%FPR | v1.3 |
|-------|--------|-----------|------|
| identity | lsb | 0.140 | 0.060 |
| identity | dct | 0.980 | 0.980 |
| identity | dwt | 1.000 | 1.000 |
| jpeg_70 | dct | 0.860 | 0.840 |
| jpeg_70 | dwt | 0.880 | 0.880 |
| jpeg_50 | dct | 0.460 | 0.460 |
| jpeg_50 | dwt | 0.580 | 0.600 |
| jpeg_50 | spread | 0.520 | 0.520 |
| jpeg_50 | tree_ring | 0.920 | 0.920 |

Cover mean identity 0.557, jpeg_70 0.515, jpeg_50 0.493.
Lock threshold **0.756**. Nested OR FPR 0.12 vs legacy 0.06. Default
`legacy`. Default `present` stays ~0.67. BSDS camera FPR at 0.67 is
**0.08** (gate was <= 0.12).

JPEG50 DCT did not lift without a recook. High-Q identity LSB ticked
0.06 -> 0.14 because Q95 no longer takes a 0.5 FSNet blend. Stage D cook
stays held.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67. File not overwritten.
- Patchwork weight 0. No UniFreq. No remover. No from-scratch Q50.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
