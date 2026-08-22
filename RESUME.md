# Resume (v1.0.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v1.0.0 production cut.
Public lamp: https://veilscan.jonbailey.xyz/

## Where we stopped

- Generator lock ~0.67 stays default `present` (BSDS FPR 0.66, DIV2K 0.72).
- Camera sidecar `op-v1.0.0-camera-locked-n50` ~0.758.
- JPEG-hardened FSNet + jpeg_like blend. Ensemble JPEG70 DCT TPR 0.64.
- LOAO holdout-dct proves FSNet is a frequency specialist.
- Patchwork unsupported. Fusion `legacy`.
- 2026-08-22 polish: README/ARCHITECTURE/LIMITATIONS/fetch UA, lamp cache
  1.3.0, GitHub homepage + description, tweet-ready v1.0 copy. No science
  recook. No present/OR flip.

## Next

Do not flip default `present`. Camera JPEG70 DWT TPR is still weak (0.30).
Optional: more JPEG50 coverage, or a true already-JPEG detector so more
camera stills take the blend. Not another YAML weight on patchwork.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera`
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
