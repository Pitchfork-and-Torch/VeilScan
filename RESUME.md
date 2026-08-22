# Resume (v1.7.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.7.0 native windows.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Field JPEG stack A+B+C+D shipped (blend, policy, confirmation, native 128 windows).
- 256px identity DCT TPR 0.06 -> 1.00. Generator 0.67 FPR 0.02 on that slice.
- Canonical camera OP stays 128px v1.4 ~0.756.
- Q50 recooks from v1.3 rejected. Production FSNet unchanged.
- Default `present` stays ~0.67.

## Next

JPEG50 DCT still 0.46 at 128px. Do not replace 128px lock with 256px.
Do not flip default `present`. DIV2K is confirmation only.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- From-scratch Q50 cooks
- Remover / UniFreq / Gradio from Grok
