# VeilScan v0.2.4

FSNet was an LSB shortcut. Training **without** the LSB family, plus a high-pass residual stem (RGB + LSB + 16x blur residual), made it a frequency specialist.

## Live n=8 mix

| family | FSNet cover | FSNet marked | delta |
|--------|-------------|--------------|-------|
| lsb | 0.001 | 0.002 | 0 |
| dct | 0.001 | 0.910 | +0.91 |
| spread | 0.001 | 0.947 | +0.95 |
| dwt | 0.001 | 0.980 | +0.98 |
| tree_ring | 0.001 | 0.977 | +0.98 |
| patchwork | 0.001 | 0.010 | +0.01 |

JPEG Q70 on DCT (n=6 photo): cover 0.10, marked 0.84, delta +0.74.

ResidualCNN remains the LSB head (v0.2.3). Together they cover the AWPD split: bit-plane vs frequency.

## Train

```powershell
py -3 scripts\train_lite.py --fresh --only fsnet_lite --steps 500 --families dct,spread,dwt,tree_ring --jpeg-prob 0.15 --cover-style mix
```

Do **not** mix `lsb` into this cook or the frequency cue collapses.
