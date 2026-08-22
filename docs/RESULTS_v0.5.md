# VeilScan v0.5.0

Current numbers. Generator-photo bench, not ImageNet, not UniFreq-100K.

Package: `0.5.0`. Scan JSON `schema_version` 2. Locked operating point
`op-v0.4.0-locked-n50` (named at lock; still the live cut).

## What shipped since v0.3

- Keyless decode + scanline tokens + HUD report (v0.3.x)
- Frozen generator bench, `family_hint`, specialist-OR behind a flag (v0.4)
- Locked threshold ~0.67 at FPR 0.05 on that slice (v0.4.1)
- `veilscan inspect` (v0.4.1)
- Vectorized DCT/spread embeds; `veilscan bench --covers DIR` (v0.5.0)

## Locked slice (n=50, size 128, photo, identity + jpeg_70)

| slice | family | AUC | TPR@5%FPR |
|-------|--------|-----|-----------|
| identity | lsb | 1.000 | 1.000 |
| identity | dct | 1.000 | 1.000 |
| identity | dwt | 1.000 | 1.000 |
| identity | spread | 1.000 | 1.000 |
| identity | tree_ring | 0.985 | 0.980 |
| identity | patchwork | 0.964 | 0.640 |
| jpeg_70 | lsb | 0.600 | 0.100 |
| jpeg_70 | dct | 0.965 | 0.940 |
| jpeg_70 | dwt | 0.894 | 0.700 |
| jpeg_70 | spread | 1.000 | 1.000 |
| jpeg_70 | tree_ring | 1.000 | 1.000 |
| jpeg_70 | patchwork | 0.922 | 0.600 |

Cover mean identity ~0.64. Default ensemble threshold 0.67. Fusion `legacy`.

specialist-OR A/B on the same covers: FPR 0.14 vs legacy 0.05. Nested even/odd
holdout: legacy FPR 0.04, OR FPR 0.22. Default stays legacy.

Full tables: `docs/bench/latest.md`.

## Camera FPR

Not measured. Run `veilscan bench --covers DIR` on an operator-owned folder.
Results go to `docs/bench/camera.json` and must not overwrite the generator lock.
