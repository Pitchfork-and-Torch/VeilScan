# VeilScan v0.9.0

Historical patchwork decision. Current numbers: `docs/RESULTS_v1.0.md`.

Patchwork labeled **unsupported** on camera stills. Keyless permutation-null
pair-mean test does not separate the eval generator (secret pairing). YAML
weight 0 so the head cannot pollute `full_mean`.

Package: `0.9.0`. Generator lock and BSDS camera sidecar unchanged.
Default `present` still not flipped. Fusion stays `legacy`.

## Measurement (BSDS test, n=24, 128px, identity, patchwork head only)

| | cover mean | marked mean | AUC | TPR@5%FPR |
|--|------------|-------------|-----|-----------|
| patchwork head | 0.258 | 0.260 | 0.50 | 0.08 |

z_cover ~0.92, z_marked ~0.69 (null pairing is not the secret key).
Camera ensemble TPR for the patchwork family was already 0.02 (BSDS) /
0.10 (DIV2K) in v0.7-v0.8.

The NEXT_MASSIVE_UPGRADE contract was: TPR>=0.6 @ FPR 0.05 or unsupported
with weight 0. This is the second branch. A keyed or trained specialist
would be a new model, not a YAML bump.

## Unchanged

- Generator `op-v0.4.0-locked-n50` ~0.67
- BSDS camera `op-v0.7.0-camera-locked-n50` ~0.753
- DIV2K confirmation `op-v0.8.0-camera-div2k-locked-n50` ~0.765
- FSNet v0.7 mix cook. ResidualCNN unchanged.

## Next

Camera JPEG70 DCT, or LOAO folders. Not another patchwork YAML weight.
