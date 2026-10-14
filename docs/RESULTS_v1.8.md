# VeilScan v1.8.0

Q-table DCT inspect. No production recook. Generator lock unchanged.
BSDS 128px camera sidecar unchanged. Default `present` stays ~0.67.

Package: `1.8.0`. Schema 5 JSON adds `jpeg_luma_q_34` and `jpeg_luma_q_43`
(optional luma DQT steps at the `embed_dct` bins). Fusion unchanged.

## Why not a JPEG50 DCT TPR lift

Eval DCT marks add amplitude 14 at 8x8 bins (3,4) and (4,3). IJG Q50
luma steps at those bins are about 51 and 56. The mark sits under the
quantizer. Two Stage D recooks of v1.3 FSNet already made FSNet-head
jpeg_50 DCT TPR worse (0.40 -> 0.30 / 0.34). Contract still holds: no
from-scratch Q50, do not flip `present`, do not replace the 128px lock.

n=24 camera 128px smoke (not a lock):

- classical DCT head after jpeg_50: AUC 0.53, TPR@5% 0.08
- pair (3,4)+(4,3) energy after jpeg_50: AUC 0.60, TPR@5% 0.17
- Ensemble jpeg_50 DCT TPR at 128px remains **0.46** (v1.4 lock)

## What 1.8 is

- `inspect_jpeg` reports luma Q at (3,4) and (4,3)
- scan JSON + CLI print those steps next to quality_est
- DCT extras include `pair_34_43` (mean |C34|+|C43|). Score mix unchanged.

Use `veilscan inspect FILE.jpg` and read `luma_q(3,4)` vs the DCT pair
energy. If the step is larger than the mark, Q50 erasure is expected.

## Still true

- Generator `op-v0.4.0-locked-n50` ~0.67
- BSDS camera sidecar `op-v1.4.0-camera-locked-n50` ~0.756
- Production FSNet sha256
  `8064ca659e44bf7b832b3d847ce16a508a1b6d0c7692c90d69381f9d5b6bb195`
- No UniFreq. No remover. No from-scratch Q50. No lsb on FSNet.
