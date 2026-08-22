# Resume (v0.6.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.6.0 camera sidecar.
Public lamp: https://veilscan.jonbailey.xyz/ (mast still decode-first).

## Where we stopped

- Generator lock `op-v0.4.0-locked-n50` ~0.67 stays default `present`.
- Camera sidecar `op-v0.6.0-camera-locked-n50` ~0.817, FPR 0.05, n=50 BSDS500.
- Nested OR FPR 0.98. Fusion `legacy`. FSNet saturates on camera stills.
- `veilscan doctor` + fetch script shipped in 0.5.1.

## Next

Do not flip default `present` to camera without a second corpus.
Do not flip specialist-OR. Optional: FSNet recook on camera covers
(without `lsb`), or LOAO folders. Lamp copy can cite RESULTS_v0.6.

## Do not

- Mix LSB into the FSNet cook
- Personal photos in git
- UniFreq in git
- Remover
- Gradio from a Grok Build command
