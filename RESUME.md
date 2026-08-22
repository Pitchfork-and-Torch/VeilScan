# Resume (v0.9.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.9.0 patchwork unsupported.
Public lamp: https://veilscan.jonbailey.xyz/ (mast still decode-first).

## Where we stopped

- Generator lock ~0.67 stays default `present` (DIV2K FPR 0.83).
- BSDS camera sidecar ~0.753 is still scan JSON `camera`.
- Patchwork is unsupported on camera (perm-null AUC 0.50). YAML weight 0.
- Fusion `legacy`. FSNet still the v0.7 mix cook.

## Next

Camera JPEG70 DCT, or LOAO folders. Do not revive patchwork with a YAML
weight. Do not flip default `present` or specialist-OR.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera` (leaks the BSDS bench)
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
