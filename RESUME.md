# Resume (v1.2.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.2.0 Q50-aware FSNet.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Fine-tuned v1.0 FSNet (not from scratch). Q50 DWT/spread/tree-ring up.
- JPEG50 DCT ensemble TPR 0.18 -> 0.32. FSNet-head DCT at Q50 still 0.26.
- Camera sidecar `op-v1.2.0-camera-locked-n50` ~0.484 (below generator lock).
- Generator lock ~0.67 FPR on BSDS camera 0.01. Not flipped.
- `--jpeg-attacks` on train_lite.py. Fusion `legacy`. Patchwork weight 0.

## Next

JPEG50 DCT is still the hole (ensemble TPR 0.32, head 0.26). Optional:
longer fine-tune or a DCT-only Q50 specialist. Do not flip default
`present`. Not another YAML weight on patchwork. DIV2K sidecar still v1.0
(PNG pack).

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
- From-scratch Q50 cooks (identity DCT collapses)
