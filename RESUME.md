# Resume (v1.6.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.6.0 confirmation.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Field JPEG stack A+B+C shipped (blend, policy, confirmation).
- DIV2K PNG sidecar ~0.796. Generator 0.67 FPR 0.59 on that PNG pack.
- BSDS 256px TPR collapsed. Canonical camera OP stays 128px v1.4 ~0.756.
- LOAO holdout-dct still collapses (0.98 -> 0.08). Cooks still allowed.
- Default `present` stays ~0.67.

## Next

Optional: JPEG50 DCT still 0.46. Do not replace 128px lock with 256px.
Do not flip default `present`. DIV2K is confirmation only.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- From-scratch Q50 cooks
- Remover / UniFreq / Gradio from Grok
