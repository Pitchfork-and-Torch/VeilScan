# Field JPEG stack (done through Stage C)

**Status:** A v1.4 blend, B v1.5 `--policy`, C v1.6 confirmation SHIPPED.

- DIV2K PNG: generator FPR 0.59 at 0.67. Sidecar ~0.796. Confirmation only.
- BSDS 256px: TPR collapsed. Canonical lock stays 128px v1.4 ~0.756.
- LOAO holdout-dct: DCT TPR 0.98 -> 0.08. FSNet is still a specialist.

## Optional leftover

JPEG50 DCT ensemble TPR 0.46. Stage D cook only with the v1.3 rails
(resume weights, no from-scratch Q50, camera FPR at 0.67 <= 0.12).

Do not flip default `present`. Do not replace 128px BSDS lock with 256px.
