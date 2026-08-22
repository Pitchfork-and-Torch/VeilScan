# VeilScan v0.8.0

Second camera corpus. DIV2K valid-HR (n=50, 128px, identity + jpeg_70).
Not BSDS500. Not UniFreq. Extracted PNGs stay gitignored.

Package: `0.8.0`. Confirmation lock: `op-v0.8.0-camera-div2k-locked-n50`.
Scan JSON `camera` sidecar is still BSDS `op-v0.7.0-camera-locked-n50`.
Default `present` is still the generator lock. Fusion stays `legacy`.

## Why this cut

v0.7 said do not flip default `present` without a second corpus. BSDS500
test cover mean was ~0.71, so the 0.67 generator lock false-fires there.
That could have been a BSDS quirk. DIV2K valid-HR is 2K NTIRE stills from
a different source.

## FPR at existing locks (DIV2K covers, identity + jpeg_70, n=100 scores)

| lock | id | threshold | FPR |
|------|----|-----------|-----|
| generator | `op-v0.4.0-locked-n50` | 0.670 | **0.83** |
| BSDS camera | `op-v0.7.0-camera-locked-n50` | 0.753 | 0.12 |
| DIV2K (this slice) | `op-v0.8.0-camera-div2k-locked-n50` | 0.765 | 0.05 |

Cover mean ~0.715 (BSDS identity was ~0.709). Same ballpark.

**Do not flip default `present`.** Two public stills packs both sit near 0.71.
The generator lock is not a camera operating point.

The BSDS sidecar (0.753) is slightly optimistic on DIV2K (FPR 0.12 vs 0.05).
Scan JSON still uses BSDS. DIV2K is confirmation, not a silent swap.

## Locked DIV2K slice (n=50, size 128, identity + jpeg_70)

| slice | family | AUC | TPR@5%FPR |
|-------|--------|-----|-----------|
| identity | lsb | 0.980 | 1.000 |
| identity | dct | 0.910 | 0.720 |
| identity | dwt | 0.951 | 0.940 |
| identity | spread | 0.969 | 0.980 |
| identity | tree_ring | 0.924 | 0.860 |
| identity | patchwork | 0.614 | 0.100 |
| jpeg_70 | lsb | 0.473 | 0.060 |
| jpeg_70 | dct | 0.611 | 0.240 |
| jpeg_70 | dwt | 0.810 | 0.720 |
| jpeg_70 | spread | 0.695 | 0.320 |
| jpeg_70 | tree_ring | 0.922 | 0.820 |
| jpeg_70 | patchwork | 0.509 | 0.100 |

In-sample OR FPR 0.13 vs legacy 0.05. Nested OR 0.12 vs legacy 0.06.
Default stays legacy.

Frequency identity still separates (weaker than BSDS DCT 0.96). Patchwork
stays unsupported. JPEG70 LSB stays dead.

## Fetch

```
py -3 scripts\fetch_camera_covers.py --manifest configs\camera_div2k_covers.manifest.json --out data\covers\camera-div2k
py -3 -m veilscan bench --covers data\covers\camera-div2k --corpus-id camera-div2k --n 50 --attacks identity,jpeg_70
```

`--corpus-id camera` plus `--write-operating-point` is refused unless
`--covers` is the frozen BSDS test folder. Archive sha256 is pinned.

## Still open

- Default `present` stays generator ~0.67.
- Patchwork on camera stills.
- Camera JPEG70 DCT.
- specialist-OR.
- No UniFreq. No remover. No personal photos in git.
