# VeilScan v1.0.0

First production cut: JPEG-hardened frequency specialist, JPEG-aware
ensemble blend, measured camera operating point, LOAO proof that FSNet is
not a generic watermark net. Keyless decode and generator lock unchanged.
No remover.

Package: `1.0.0`. Scan JSON `schema_version` 3 (`jpeg_like`, `jpeg_blockiness`).
Camera sidecar: `op-v1.0.0-camera-locked-n50` ~0.758.
FSNet sha256: `7ead6ac360f3dcce96ca9138bd988c9686988dd7bab4b8c9f555627a7af070f9`.
ResidualCNN unchanged.

## What 1.0 is

A local forensic tool that answers: is an invisible watermark present, and
if the payload is keyless plaintext, what does it say? `present` is a
measured generator-photo cut (~0.67). Camera stills get a sidecar cut
(~0.758). JPEG Q70 frequency marks are no longer drowned by hot classical
heads: the ensemble blends toward FSNet when the array looks JPEG-like.

## Why this is not another 0.x sidecar

v0.7 recooked FSNet on camera-train but the JPEG70 probe scored unattacked
covers. Honest JPEG70 camera DCT TPR@5%FPR for that net was 0.58. v1.0
recooks with `jpeg_prob` 0.7 and attacks both cover and marked in the probe.
Honest FSNet JPEG70 DCT TPR@5%FPR is **0.75**. Ensemble JPEG70 DCT TPR
moves 0.44 -> **0.64** (BSDS n=50). Cover mean on JPEG70 drops 0.70 -> 0.58.

LOAO: FSNet trained with `--holdout-family dct` falls to DCT TPR@5%FPR 0.06
vs production 0.94. Frequency specialist is real.

## Locked BSDS camera slice (n=50, 128px, identity + jpeg_70)

| slice | family | AUC | TPR@5%FPR |
|-------|--------|-----|-----------|
| identity | lsb | 1.000 | 1.000 |
| identity | dct | 0.984 | 0.840 |
| identity | dwt | 0.974 | 0.900 |
| identity | spread | 0.999 | 0.980 |
| identity | tree_ring | 0.970 | 0.820 |
| identity | patchwork | 0.649 | 0.020 |
| jpeg_70 | lsb | 0.498 | 0.060 |
| jpeg_70 | dct | 0.915 | 0.640 |
| jpeg_70 | dwt | 0.923 | 0.300 |
| jpeg_70 | spread | 0.977 | 0.860 |
| jpeg_70 | tree_ring | 0.957 | 0.700 |
| jpeg_70 | patchwork | 0.504 | 0.060 |

Cover mean identity ~0.705, jpeg_70 ~0.584. Combined threshold **0.758**
(FPR 0.05). `t_freq` 0.771 (v0.7 was 1.0, v0.9 was 0.875). Nested OR FPR
0.20 vs legacy 0.12. Default stays `legacy`. Default `present` stays the
generator lock (FPR 0.66 at 0.67 on this slice).

## DIV2K confirmation

`op-v1.0.0-camera-div2k-locked-n50` ~0.777. Generator 0.67 FPR 0.72.
BSDS 0.758 FPR 0.07. JPEG70 DCT TPR@5%FPR 0.32 (weaker than BSDS). Do not
flip default `present`.

## JPEG-aware blend

When `jpeg_like` (8x8 blockiness >= 1.10), ensemble score is
`0.5 * legacy_mix + 0.5 * fsnet`. PNG / identity camera stills stay on
legacy mix so spatial LSB is not diluted. JSON: `jpeg_like`, `jpeg_blockiness`.

## LOAO (not the default scan)

```
py -3 scripts\train_lite.py --only fsnet_lite --holdout-family dct --families dct,spread,dwt,tree_ring --covers data\covers\camera-train --cover-mix 0.35 --jpeg-prob 0.7 --out checkpoints\loao --fresh --steps 400
```

Checkpoints under `checkpoints/loao/` stay gitignored. Default scan uses
`checkpoints/fsnet_lite.pt`.

## Still true

- Patchwork unsupported (weight 0).
- JPEG70 LSB dead.
- No UniFreq. No remover. No personal photos in git.
- Do not train FSNet with `lsb` or on the frozen BSDS test pack.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
