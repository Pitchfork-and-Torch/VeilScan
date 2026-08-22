# Resume (v0.7.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.7.0 FSNet camera-train mix.
Public lamp: https://veilscan.jonbailey.xyz/ (mast still decode-first).

## Where we stopped

- Generator lock `op-v0.4.0-locked-n50` ~0.67 stays default `present`.
- Camera sidecar `op-v0.7.0-camera-locked-n50` ~0.753, FPR 0.05, n=50 BSDS500 test.
- FSNet recooked on BSDS500 train-200 + 40% synthetic photo, no lsb.
- Frozen test pack is never the cook. `t_freq` 0.875 (was 1.0).
- In-sample OR FPR 0.15 vs legacy 0.05. Fusion `legacy`.
- ResidualCNN unchanged. Patchwork on camera stills still dead.

## Next

Do not flip default `present` to camera without a second corpus.
Do not flip specialist-OR (in-sample FPR still worse). Optional: second
camera corpus, patchwork specialist, or LOAO folders.

## Do not

- Mix LSB into the FSNet cook
- Train on `data/covers/camera` (leaks the bench)
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
