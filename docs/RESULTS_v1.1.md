# VeilScan v1.1.0

Native JPEG path. Presence ensemble plus keyless plaintext decode.
No remover. Generator lock unchanged.

Package: `1.1.0`. Scan JSON `schema_version` 4 (`jpeg_container`,
`jpeg_quality_est`, `jpeg_subsampling`, plus schema 3 `jpeg_like` /
`jpeg_blockiness`).
Camera sidecar: `op-v1.1.0-camera-locked-n50` ~0.739.
FSNet and ResidualCNN weights unchanged.

## What 1.1 is

v1.0 blended toward FSNet only when the **decoded array** looked blocky
(threshold 1.10). BSDS camera JPEGs are Q95 / 4:2:0. Full-res blockiness
sits ~1.00-1.08, so `veilscan scan photo.jpg` never took the blend.

v1.1 inspects the file bytes: SOI, luma DQT quality, chroma subsampling.
`jpeg_like` is true if the file is a JPEG container **or** the array is
blocky. PNG / BMP / TIFF stay pixel-only so a clean raster is not pulled
toward FSNet.

## Locked BSDS camera slice (n=50, 128px)

Lock slice stays identity + jpeg_70. jpeg_50 is measured, not in the cut.

| slice | family | AUC | TPR@5%FPR | v1.0 TPR |
|-------|--------|-----|-----------|----------|
| identity | lsb | 0.756 | 0.060 | 1.000 |
| identity | dct | 1.000 | 1.000 | 0.840 |
| identity | dwt | 1.000 | 1.000 | 0.900 |
| identity | spread | 1.000 | 1.000 | 0.980 |
| identity | tree_ring | 0.993 | 0.940 | 0.820 |
| identity | patchwork | 0.557 | 0.060 | 0.020 |
| jpeg_70 | lsb | 0.492 | 0.060 | 0.060 |
| jpeg_70 | dct | 0.940 | 0.720 | 0.640 |
| jpeg_70 | dwt | 0.968 | 0.800 | 0.300 |
| jpeg_70 | spread | 0.980 | 0.860 | 0.860 |
| jpeg_70 | tree_ring | 0.982 | 0.900 | 0.700 |
| jpeg_70 | patchwork | 0.547 | 0.040 | 0.060 |
| jpeg_50 | lsb | 0.497 | 0.060 | n/a |
| jpeg_50 | dct | 0.743 | 0.180 | n/a |
| jpeg_50 | dwt | 0.828 | 0.420 | n/a |
| jpeg_50 | spread | 0.834 | 0.300 | n/a |
| jpeg_50 | tree_ring | 0.972 | 0.800 | n/a |
| jpeg_50 | patchwork | 0.514 | 0.060 | n/a |

Cover mean identity 0.416 (v1.0 ~0.705), jpeg_70 0.520, jpeg_50 0.558.
Combined lock threshold **0.739** (FPR 0.05). Nested OR FPR 0.20 vs
legacy 0.10. Default stays `legacy`. Default `present` stays the
generator lock.

## The FPR hole moved

At generator 0.67, BSDS camera FPR is **0.07** (v1.0 was 0.66). The lock
did not flip. JPEG containers pull unmarked scores toward a quiet FSNet,
so camera stills no longer sit on top of the generator cut.

Identity LSB TPR on those rasters falls 1.00 -> 0.06. That is the blend
doing what it says: a JPEG file is a frequency object. Spatial LSB that
exists only in the decoded raster of a JPEG is not a field path (re-save
as JPEG kills it). PNG LSB is unchanged (no container, no blend).

JPEG70 DWT TPR 0.30 -> **0.80** without recooking FSNet. The leftover
v1.0 hole was the blend not firing, not the net.

## JPEG50 (measured, not locked)

Tree-ring still separates (TPR@5%FPR 0.80). DCT 0.18 / DWT 0.42 / spread
0.30. A Q50 recook is the next science, not another YAML weight.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67. File not overwritten.
- DIV2K sidecar remains `op-v1.0.0-camera-div2k-locked-n50` (PNG pack).
- Patchwork unsupported (weight 0).
- No UniFreq. No remover. No personal photos in git.
- Do not train FSNet with `lsb` or on the frozen BSDS test pack.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
