# Resume (after v0.3.0 decode)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.3.0 keyless plaintext decode.
Public lamp: https://veilscan.jonbailey.xyz/ site v1.1.0.

## Where we stopped

v0.3.0 shipped `veilscan decode` / `embed-text` plus in-tab LSB / PNG text / JPEG COM read.

Complementary deep heads still work:

- ResidualCNN = LSB
- FSNet-lite = frequency
- Never train FSNet with `lsb` in `--families`

Next science contract: `docs/NEXT_MASSIVE_UPGRADE.md` (v0.4 operating point).

## First session on the bench (M1, not a new net)

1. Desk-check `veilscan` then claim.
2. Implement `scripts/bench.py` / `veilscan bench` from M1 (start n=20 if short, protocol n>=50).
3. Freeze `configs/bench_protocol.yaml` + write `docs/bench/latest.json`.
4. Do not enable `calibration.fitted.json` as default until the bench exists.

## Do not

- Mix LSB into the FSNet cook
- Strip LSB planes off ResidualCNN
- Remover / SynthID clone / UniFreq in git
- Gradio from a Grok Build command
