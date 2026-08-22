# Resume (v1.3.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.3.0 DCT-Q50 FSNet.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Lamp copy matches CLI: inspect, JPEG SOI/Q-table/subsampling, ResidualCNN vs FSNet, doctor, schema 4, JPEG+PNG drop.
- DCT-heavy Q50 fine-tune of v1.2 weights. Ensemble JPEG50 DCT 0.32 -> 0.46.
- Camera sidecar `op-v1.3.0-camera-locked-n50` ~0.758.
- Generator lock ~0.67 FPR on BSDS camera 0.08. Not flipped.
- Fusion `legacy`. Patchwork weight 0.

## Next

JPEG50 DCT is still incomplete (0.46). Optional: another fine-tune or
leave it labeled weak. Do not flip default `present`. Not another YAML
weight on patchwork. DIV2K sidecar still v1.0 (PNG pack).

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
- From-scratch Q50 cooks
