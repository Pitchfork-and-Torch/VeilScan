# VeilScan v0.2.2

CUDA is live: `torch 2.13.0+cu130` on RTX 4080.

## What was broken

The old ResidualCNN averaged 5x5 residuals in 0-1 RGB. LSB is a 1/255 perturbation, so every image collapsed to the same logit (~0.5). Overfit on 8 images failed until an **explicit LSB bit-plane** was concatenated as extra channels.

## ResidualCNN (shipped)

- Input: RGB (0-1) + 3 LSB planes.
- Train: 400 steps, batch 32, CUDA, families `lsb,dct,spread`, mix covers, JPEG p=0.2.
- Val AUC **0.840**.
- Live n=8 mix pair means:

| family | cover | marked | delta |
|--------|-------|--------|-------|
| lsb | 0.428 | 0.964 | +0.535 |
| dct | 0.428 | 0.648 | +0.219 |
| spread | 0.428 | 0.624 | +0.196 |
| tree_ring | 0.428 | 0.611 | +0.183 |
| dwt | 0.428 | 0.510 | +0.081 |

`residual_cnn` is now in `peak_ok`.

## FSNet-lite

400-step CUDA train on dct/spread/dwt still sat at loss ~0.69, val AUC 0.55. Checkpoint **deleted** so the plugin stays skipped. Needs a bit-plane or stronger frequency stem before another cook.

## Still later

JPEG-robust FSNet, DDIM inversion, UniFreq, TPR@1% FPR on photographs.
