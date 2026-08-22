# Resume (v1.1.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.1.0 native JPEG path.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Native JPEG inspect (SOI / DQT quality / subsampling). Schema 4.
- `jpeg_like` = container or blockiness >= 1.10.
- Camera sidecar `op-v1.1.0-camera-locked-n50` ~0.739.
- Generator lock ~0.67 FPR on BSDS camera 0.66 -> 0.07. Not flipped.
- JPEG70 DWT TPR 0.30 -> 0.80 without recook.
- JPEG50 measured. Fusion `legacy`. Patchwork weight 0.

## Next

JPEG50 frequency TPR is still weak (DCT 0.18, DWT 0.42, spread 0.30).
Optional: a Q50-aware FSNet recook. Do not flip default `present`.
Not another YAML weight on patchwork. DIV2K sidecar still v1.0 (PNG pack).

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
