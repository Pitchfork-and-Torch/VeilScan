# Resume (v1.8.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.8.0 Q-table DCT inspect.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Field JPEG stack A+B+C+D shipped (blend, policy, confirmation, native 128 windows).
- v1.8 inspect prints luma Q at DCT bins (3,4) and (4,3). Score mix unchanged.
- 256px identity DCT TPR 0.06 -> 1.00. Generator 0.67 FPR 0.02 on that slice.
- Canonical camera OP stays 128px v1.4 ~0.756.
- Q50 recooks from v1.3 rejected. Production FSNet unchanged.
- Default `present` stays ~0.67.
- JPEG50 DCT leftover is physical: eval amp 14 vs Q50 steps ~51/56.

## Next

JPEG50 DCT still 0.46 at 128px. Cooks only if a probe beats 0.40 FSNet-head
TPR@5% under v1.3 rails. Do not replace 128px lock with 256px.
Do not flip default `present`. DIV2K is confirmation only.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- From-scratch Q50 cooks
- Remover / UniFreq / Gradio from Grok
