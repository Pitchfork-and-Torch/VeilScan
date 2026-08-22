# Next upgrade: v1.2 Q50-aware FSNet

**Status:** SHIPPED in v1.2.0. Next leftover: JPEG50 DCT (ensemble TPR 0.32, FSNet head 0.26).
v1.1.0 native JPEG path shipped
(`docs/RESULTS_v1.1.md`). Generator `present` stays locked.

**Identity:** presence ensemble plus keyless plaintext decode. No remover.
No UniFreq in git. ResidualCNN unchanged. Patchwork weight 0.

Lamp: https://veilscan.jonbailey.xyz/

---

## 1. Why this upgrade

v1.1 made camera JPEGs take the FSNet blend. JPEG70 DWT TPR@5%FPR moved
0.30 -> 0.80 **without** a recook. JPEG50 did not: DCT 0.18, DWT 0.42,
spread 0.30. Tree-ring still 0.80. The net was cooked with `jpeg_prob` 0.7
and equal `jpeg_90` / `jpeg_70` / `jpeg_50`, so Q50 is only ~23% of
samples. Harder JPEG is under-seen, not missing from the file parser.

**North star for v1.2:** a candidate FSNet that raises JPEG50 frequency
TPR on the frozen BSDS test pack without wrecking JPEG70 DWT (~0.80) or
the generator lock. Default `present` does not flip.

## 2. Do not break

- Generator lock `op-v0.4.0-locked-n50` ~0.67. Never overwrite from `--covers`.
- FSNet cook: no `lsb`. No frozen BSDS test pack (`data/covers/camera`).
- ResidualCNN weights stay `e82851366d533e86eaee9a48c75f6acc7be1166cd191641faa013b7cba37ac79`.
- Patchwork YAML weight 0. Fusion `legacy` unless nested OR FPR also holds.
- No remover. No Gradio from a Grok Build command.
- Schema 4 fields stay. Do not bump schema for a cook.
- Probe must JPEG **covers and marked** (v0.7 leak).

## 3. Feature (this cut)

| Piece | Truth |
|-------|--------|
| `--jpeg-attacks` on `train_lite.py` | Replaces equal 90/70/50 draw |
| Candidate cook | `checkpoints/candidates/`, never production until probe wins |
| Covers | `data/covers/camera-train`, mix 0.35, families dct,spread,dwt,tree_ring |
| Probe | frozen test `data/covers/camera`, identity + jpeg_70 + jpeg_50 |
| Promote | only if JPEG50 DCT TPR@5%FPR beats production **and** JPEG70 DWT TPR does not collapse |
| Camera sidecar | re-lock n=50 as `op-v1.2.0-camera-locked-n50` if promoted |
| JPEG50 | still **not** in the OP lock slice (identity + jpeg_70) |

## 4. Acceptance

- [x] `--jpeg-attacks jpeg_50` is honored; `lsb` still exits 2
- [x] Candidate lives under `checkpoints/candidates` until probe
- [x] Honest probe (both sides JPEG) vs production on dct/dwt/spread/tree_ring
- [x] Promote only on a measured win; from-scratch Q50 cook rejected
- [x] Camera n=50 sidecar rewritten only after promote; generator lock untouched
- [x] Nested OR still not default
- [x] Pytest green
- [ ] Lamp mast 1.2.0. Site cache-bust **1.5.0**
- [ ] Latest-only private GitHub release `v1.2.0`

## 5. Not this cut

- Flip default `present`
- UniFreq-100K
- In-tab presence scores
- Patchwork keyed specialist
- Recook ResidualCNN
