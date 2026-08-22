# Resume (v0.5.0)

Private GitHub: `Pitchfork-and-Torch/VeilScan` v0.5.0 vectorized DCT/spread + camera bench adapter.
Public lamp: https://veilscan.jonbailey.xyz/ site v1.1.7.

## Where we stopped

- Generator OP stays locked: `op-v0.4.0-locked-n50` threshold 0.67, fusion `legacy`.
- `embed_dct` / `embed_spread` are batched scipy ortho DCT (matches cv2).
- `veilscan bench --covers DIR` runs the same protocol on real JPEGs. Does not
  write `configs/operating_point.json`. Nested holdout FPR is in `ab.nested_holdout`.

## Next

Point `--covers` at a camera folder of n>=50 (operator-owned, not git). Compare
camera FPR at 0.67 to the generator lock. Deep LOAO ckpts optional.

## Do not

- Mix LSB into the FSNet cook
- Strip LSB planes off ResidualCNN
- Remover / SynthID clone / UniFreq in git
- Flip specialist-OR without nested holdout
- Gradio from a Grok Build command
- Commit personal photo folders
