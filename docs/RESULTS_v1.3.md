# VeilScan v1.3.0

DCT-heavy Q50 FSNet fine-tune. Lamp copy matches the CLI: inspect, JPEG
container fields, ResidualCNN vs FSNet, decode in-tab. No remover.
Generator lock unchanged.

Package: `1.3.0`. Scan JSON `schema_version` 4.
Camera sidecar: `op-v1.3.0-camera-locked-n50` ~0.758.
FSNet sha256: `8064ca659e44bf7b832b3d847ce16a508a1b6d0c7692c90d69381f9d5b6bb195`.
ResidualCNN unchanged.

## What 1.3 is

v1.2 lifted JPEG50 DWT/spread/tree-ring. JPEG50 DCT stayed 0.32 ensemble /
0.26 FSNet-head. v1.3 fine-tunes v1.2 weights with a DCT-heavy family draw
and more Q50. A from-scratch Q50 cook stays banned.

```
py -3 scripts\train_lite.py --only fsnet_lite --steps 300 --lr 0.0002 --families dct,dct,dct,dwt,spread,tree_ring --jpeg-prob 0.8 --jpeg-attacks jpeg_50,jpeg_50,jpeg_50,jpeg_70 --covers data\covers\camera-train --cover-mix 0.35 --out checkpoints\candidates
```

Honest n=50 FSNet-head probe: dct/jpeg_50 0.26 -> **0.40**. Identity DCT
0.98 held. JPEG70 DWT 0.88 held.

## Locked BSDS camera slice (n=50, 128px)

| slice | family | AUC | TPR@5%FPR | v1.2 TPR |
|-------|--------|-----|-----------|----------|
| identity | lsb | 0.765 | 0.060 | 0.100 |
| identity | dct | 0.996 | 0.980 | 0.960 |
| identity | dwt | 1.000 | 1.000 | 1.000 |
| identity | spread | 0.999 | 1.000 | 1.000 |
| identity | tree_ring | 0.988 | 0.940 | 0.940 |
| identity | patchwork | 0.565 | 0.060 | 0.080 |
| jpeg_70 | lsb | 0.486 | 0.080 | 0.040 |
| jpeg_70 | dct | 0.964 | 0.840 | 0.760 |
| jpeg_70 | dwt | 0.972 | 0.880 | 0.880 |
| jpeg_70 | spread | 0.984 | 0.920 | 0.900 |
| jpeg_70 | tree_ring | 0.992 | 0.980 | 0.980 |
| jpeg_70 | patchwork | 0.542 | 0.040 | 0.060 |
| jpeg_50 | lsb | 0.495 | 0.080 | 0.040 |
| jpeg_50 | dct | 0.772 | 0.460 | 0.320 |
| jpeg_50 | dwt | 0.861 | 0.600 | 0.560 |
| jpeg_50 | spread | 0.848 | 0.520 | 0.480 |
| jpeg_50 | tree_ring | 0.972 | 0.920 | 0.940 |
| jpeg_50 | patchwork | 0.508 | 0.120 | 0.060 |

Cover mean identity 0.410, jpeg_70 0.505, jpeg_50 0.550.
Lock threshold **0.758** (FPR 0.05). Nested OR FPR 0.12 vs legacy 0.06.
Default stays `legacy`. Default `present` stays ~0.67.

At generator 0.67, BSDS camera FPR is **0.08** (v1.2 was 0.01; v1.0 was
0.66). Quieter covers from v1.2 were partly spent to buy DCT-Q50 TPR.

JPEG50 DCT 0.32 -> **0.46**. Still not a solved family. The lamp says so.

## Lamp

https://veilscan.jonbailey.xyz/ lists inspect, JPEG container inspect,
ResidualCNN vs FSNet, decode families, doctor, and honest Q50 DCT limits.
Decode still in-tab only. Presence still CLI.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67. File not overwritten.
- DIV2K sidecar remains v1.0 PNG pack.
- Patchwork weight 0. No UniFreq. No remover.
- Do not train FSNet with `lsb` or on frozen BSDS test. No from-scratch Q50.

## Generator lock (unchanged)

`op-v0.4.0-locked-n50` ~0.67. See `docs/RESULTS_v0.5.md`.
