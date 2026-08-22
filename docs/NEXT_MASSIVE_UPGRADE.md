# Field JPEG stack after v1.4.0

**Status:** Stage A SHIPPED (v1.4.0 quality-aware blend, schema 5).
JPEG50 DCT TPR still 0.46. Identity LSB 0.14 (was 0.06).

## Remaining stages

- **B v1.5:** `--policy generator|camera|both` (default generator).
- **C v1.6:** DIV2K re-lock, 256px BSDS confirmation, LOAO holdout-dct.
- **D held:** DCT-Q50 cook only if needed; A did not lift 0.46.

Rails unchanged: do not flip default `present`, no from-scratch Q50, no
lsb on FSNet, no frozen test pack, no remover.
