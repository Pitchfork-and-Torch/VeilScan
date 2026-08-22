# Resume (v0.4.0 slice A)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.4.0 generator bench + family_hint.
Public lamp: https://veilscan.jonbailey.xyz/ site v1.1.5.

## Where we stopped

Slice A landed: `veilscan bench`, `configs/bench_protocol.yaml`, schema 2 `family_hint`,
specialist-OR behind `fusion.mode` (default still `legacy`), FSNet `--families lsb` exits 2.

Operating point is `provisional` until `veilscan bench --n 50 --write-operating-point`.

Complementary deep heads:

- ResidualCNN = LSB (keep bit planes)
- FSNet-lite = frequency (never pass `lsb` in `--families`)

## Next session

1. Run n=20 identity+jpeg_70 if `docs/bench/latest.json` is missing, then n=50 lock.
2. A/B `specialist_or` vs `legacy` on that JSON; flip default only if FPR does not rise.
3. Slice B: lamp scanlines + token scorer; `veilscan inspect`.

## Do not

- Mix LSB into the FSNet cook
- Strip LSB planes off ResidualCNN
- Remover / SynthID clone / UniFreq in git
- Gradio from a Grok Build command
