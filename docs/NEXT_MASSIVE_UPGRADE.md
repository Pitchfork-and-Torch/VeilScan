# Field JPEG stack (done through Stage D)

**Status:** A v1.4 blend, B v1.5 `--policy`, C v1.6 confirmation,
D v1.7 native 128 windows SHIPPED.

- DIV2K PNG: generator FPR 0.59 at 0.67. Sidecar ~0.796. Confirmation only.
- BSDS 256px: v1.6 TPR collapsed because deep heads downscaled the whole
  plate. v1.7 native 128 windows restore identity DCT TPR 1.00 at 256px.
  Canonical lock stays 128px v1.4 ~0.756.
- LOAO holdout-dct: DCT TPR 0.98 -> 0.08. FSNet is still a specialist.
- Q50 recook from v1.3 weights rejected (FSNet-head jpeg_50 DCT 0.40 ->
  0.30 / 0.34). Production weights unchanged.

## Leftover

JPEG50 DCT ensemble TPR at 128px is still 0.46. v1.8.0 exposes luma Q at
the DCT mark bins so the leftover is inspectable: eval amp 14 vs Q50
steps ~51/56. Further cooks only with the v1.3 rails (resume weights, no
from-scratch Q50, camera FPR at 0.67 <= 0.12) and only if a probe beats
0.40 FSNet-head TPR@5%.

Do not flip default `present`. Do not replace 128px BSDS lock with 256px.
Do not flip fusion to `specialist_or`.
