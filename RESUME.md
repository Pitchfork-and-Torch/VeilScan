# Resume (v0.8.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.8.0 DIV2K second corpus.
Public lamp: https://veilscan.jonbailey.xyz/ (mast still decode-first).

## Where we stopped

- Generator lock `op-v0.4.0-locked-n50` ~0.67 stays default `present`.
- BSDS camera sidecar `op-v0.7.0-camera-locked-n50` ~0.753 is still scan JSON `camera`.
- DIV2K valid-HR confirmation `op-v0.8.0-camera-div2k-locked-n50` ~0.765.
- FPR at 0.67 on DIV2K: 0.83. FPR at BSDS 0.753 on DIV2K: 0.12.
- Two public stills packs agree: do not flip default `present`.
- Fusion `legacy`. FSNet still the v0.7 mix cook. ResidualCNN unchanged.

## Next

Patchwork on camera stills, camera JPEG70 DCT, or LOAO folders.
Do not flip specialist-OR. Do not flip default `present` without a third
corpus that is actually cool at 0.67 (none so far).

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera` (leaks the BSDS bench)
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
