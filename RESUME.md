# Resume (v1.4.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.4.0 quality-aware blend.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Schema 5 `jpeg_freq_weight` from JPEG quality. No FSNet recook.
- Camera sidecar `op-v1.4.0-camera-locked-n50` ~0.756.
- Generator lock ~0.67 FPR on BSDS 0.08. JPEG50 DCT TPR 0.46 held.
- Field JPEG stack Stage A shipped. Next: Stage B `--policy`.

## Next

Stage B: `scan|inspect --policy generator|camera|both` (default generator).
Then Stage C: DIV2K re-lock, 256px confirmation, LOAO. Stage D cook only
if DCT-Q50 needs it. Do not flip default `present`.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- From-scratch Q50 cooks
- Remover / UniFreq / Gradio from Grok
