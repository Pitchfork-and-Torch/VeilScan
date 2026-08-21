# Resume (parked 2026-08-21)

Private GitHub: `Pitchfork-and-Torch/VeilScan` @ `313aaf4` (plan) on top of `654dcf9` (v0.2.4).

## Where we stopped

Complementary deep heads work:

- ResidualCNN = LSB (cover ~0.08, marked ~0.99)
- FSNet-lite = frequency (DCT/spread/DWT/Tree-Ring ~+0.9; JPEG70 DCT +0.74)
- Never train FSNet with `lsb` in `--families`

Next contract: `docs/NEXT_MASSIVE_UPGRADE.md` (v0.2.4 -> v0.3 operating point).

## First session tomorrow (M1, not a new architecture)

1. Desk: `py -3 $env:USERPROFILE\.grok\desk\desk.py check $env:USERPROFILE\veilscan` then claim.
2. Implement `scripts/bench.py` / `veilscan bench` from M1 (start n=20 if short, protocol n>=50).
3. Freeze `configs/bench_protocol.yaml` + write `docs/bench/latest.json`.
4. Fix README: deep plugins are **on**; kill "silent until trained."
5. Do not enable `calibration.fitted.json` as default until the bench exists.

## Do not

- New detector before the bench
- Mix LSB into the FSNet cook
- Strip LSB planes off ResidualCNN
- Remover / SynthID clone / UniFreq in git
- Gradio from a Grok Build command
