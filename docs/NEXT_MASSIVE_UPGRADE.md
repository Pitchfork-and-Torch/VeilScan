# Operating-point contract (v0.4) - SHIPPED

**Status:** historical. Landed across v0.4.0 / v0.4.1. Current package is
**v1.0.0** (`docs/RESULTS_v1.0.md`). JPEG-hardened FSNet + jpeg-aware blend.
Camera sidecar ~0.758. Patchwork unsupported. Generator lock stays default
`present`. Fusion still `legacy` (OR nested FPR worse). Next science is
JPEG70 DWT on camera, or an already-JPEG detector so more stills take the
blend.

**Baseline when written:** v0.3.0 keyless decode.  
**Identity:** presence ensemble plus keyless plaintext decode. No remover.
No SynthID verifier. No mandatory cloud weights.

`docs/UPGRADE_PLAN.md` stays as the historical v0.1 map.

---

## 1. Why this upgrade, not another detector

v0.2.4 already has the architecture the original brief asked for in spirit:

- Plugin ensemble, YAML `peak_ok`, skip-without-ckpt.
- Classical LSB + frequency heads.
- **ResidualCNN = LSB specialist** (cover ~0.08, marked ~0.99).
- **FSNet-lite = frequency specialist** (DCT/spread/DWT/Tree-Ring deltas ~+0.9;
  JPEG Q70 DCT still ~+0.74). Must not train FSNet on LSB or the cue collapses.
- 42 tests, CUDA cook, private GitHub.

What it does *not* have is an **operating point**: a threshold you can defend
on images that are not 128px sine/photo *generators*. Scores are still
heuristic sigmoids. Default calibration is identity. README still says deep
plugins are off. Patchwork is a hole. Fusion is a peak of mixed classical +
deep heads, so a noisy classical extra can still dominate a clean 0.001 FSNet
cover.

**North star for v0.4:** a forensic user scans a megapixel JPEG, gets
`present` plus a **family hint** (`lsb` | `frequency` | `unknown`), and the
false-positive rate at that decision is *measured* on a frozen bench, not
guessed from n=8.

If we add ten more plugins before that bench, we will re-tune `peak_ok` in
the dark again.

---

## 2. Honest baseline (do not regress)

### Works

| Capability | Evidence |
|------------|----------|
| Scan CLI / JSON `schema_version` | v0.2.0+ |
| Complementary deep split | RESULTS_v0.2.3 (LSB) + RESULTS_v0.2.4 (frequency) |
| JPEG Q70 DCT still separates on FSNet | n=6 photo, delta +0.74 |
| CUDA train loop | RTX 4080, `torch 2.13.0+cu130`, `train_lite.py --families --fresh --cover-style` |
| LSB-plane lesson | ResidualCNN overfit failed until explicit bit planes |

### Known lies / thin ice

| Item | Truth |
|------|--------|
| README synthetic table | Written when deep nets were off. Stale. |
| Threshold 0.48 | Chosen on small-n sine selftest, not FPR@1% |
| `calibration.fitted.json` | Exists, **not** default. Enabling it without a bench will move pytest. |
| FSNet val_auc 1.0 | Same generator families it trained on. Not UniFreq. |
| RS "quadratic" | Computed; **score still uses the RM/SM gap** (quadratic saturates on sines). |
| SPA | Implemented; **out of** `peak_ok` (saturates). |
| Tree-Ring | FSNet sees the *approx generator*. Not DDIM inversion. |
| Patchwork / SVD / hidden_approx | Weak or unused by deep cooks. |
| WMD prune / inversion | CLI / skip stubs. |
| Photo FPR | `style=photo` is value-noise octaves, not ImageNet/COCO. |

### Do not break

- ResidualCNN cook: **no LSB-free RGB-only stem**.
- FSNet cook: **do not pass `--families ...,lsb`**.
- Deep skip if checkpoint missing (`test_deep_skipped_without_ckpt` uses empty dir).
- Ethics: no stripper.

---

## 3. Upgrade thesis (one sentence)

Turn two specialist nets plus classical LSB/frequency into a **measured
decision rule** with a frozen bench, a family hint, and a threshold you can
quote, then spend remaining budget only on holes the bench still fails
(patchwork, social-JPEG, inversion if weights exist).

---

## 4. Phased contract (v0.4.x)

### M1 -- Frozen bench + operating point (must ship first)

**Why:** every later model change is unmeasurable without this. This is the
actual "massive" part, not more architecture.

**Build:**

- `scripts/bench.py` / `veilscan bench`
  - Covers: `sine` and `photo` generators, n>=50 each (pytest uses n=2).
  - Families: lsb, dct, dwt, spread, tree_ring, patchwork (and svd as known-miss).
  - Attacks: identity, jpeg_90, jpeg_70, jpeg_50, resize_90, crop_90, noise, blur.
  - Metrics: AUC, TPR @ FPR 0.01 and 0.05, mean cover / marked, per-detector
    and ensemble. Write `docs/bench/latest.json` + markdown table.
- **Lock a seed and a protocol file** `configs/bench_protocol.yaml` so reruns
  are comparable.
- Choose `threshold` (and optional specialist cutoffs) from the **identity +
  jpeg_70** slice. Write `configs/operating_point.json`:
  `{threshold, t_lsb, t_freq, fpr_est, n, date, protocol}`.
- Engine reads operating_point if present; tests stay on a fixture copy or
  identity so pytest does not track the bench.

**Success:**

- Bench script exits 0; JSON checked in or published as an artifact.
- Documented FPR on generator-photo identity. Target: ensemble FPR <= 0.05
  at the chosen threshold on n>=50 photo covers (not a photo corpus yet).
- TPR @ that FPR: LSB and DCT/spread/DWT/tree_ring each >= 0.80 on identity
  (deep + classical). Patchwork allowed to fail (drives M2).
- JPEG70: frequency TPR retention >= 0.70 vs identity (FSNet already hints
  this). LSB JPEG70 may collapse; **report**, do not hide.

**Files:** `scripts/bench.py`, `eval/harness.py`, `configs/bench_protocol.yaml`,
`configs/operating_point.json`, `cli.py`, `docs/LIMITATIONS.md`, `README.md`
(kill the "deep plugins off" sentence).

**Effort / risk:** medium. Risk is overfitting the generator. Mitigation:
hold out `style=photo` seeds 10000+ from train seeds; never train on bench
seeds.

### M2 -- Holes the bench will name

**Patchwork.** Last systematic miss for both deep heads (~0 delta). Options,
in order:

1. Strengthen classical `patchwork` with a permutation null on pair means
   (still keyless). Raise confidence only if n=50 bench TPR > 0.6 at FPR 0.05.
2. Tiny specialist net trained **only** on patchwork vs cover, LSB-free and
   DCT-free so it cannot cheat. Same skip-without-ckpt rule.
3. If still dead, document as unsupported. Do not leave a 0.8 YAML weight
   that does nothing.

**Vectorize generators.** `embed_dct` / `embed_spread` Python 8x8 loops are
why 500-step cooks wait on CPU data, not GPU. Rewrite with `cv2.dct` on
strided views or a batched numpy DCT. Target: 10x faster `__getitem__`.

**On-disk cover cache.** Optional `data/covers/{sine,photo}/*.png` generated
once; train and bench read files. Stops seed drift.

**Deep LOAO (real, not the classical alias):**

- Retrain ResidualCNN with `--holdout-family lsb` -> LSB TPR must drop.
  Proves it is not a generic "any watermark" head.
- Retrain FSNet with `--holdout-family dct` -> DCT TPR must drop.
- Store holdout ckpts under `checkpoints/loao/` so default scan stays on
  the production pair.

**SVD / hidden_approx.** Either a 200-step specialist or an explicit miss in
the bench table. No silent 0.5.

**Success:** patchwork either has TPR>=0.6 @ FPR 0.05 or is labeled unsupported
in LIMITATIONS with YAML weight 0. DCT embed no longer dominates train time.

### M3 -- Fusion 2.0 (specialist OR)

Current mix (0.2/0.2/0.2/0.4 peak of `peak_ok`) lets a mediocre classical
head outvote FSNet 0.001 / Residual 0.08 on a clean cover, or bury a 0.91
DCT hit if classical is quiet.

**New decision rule (v1, keep old mix behind `fusion.mode: legacy`):**

```
lsb_score  = residual_cnn   (skip -> 0)
freq_score = fsnet_lite     (skip -> 0)
class_score = peak of remaining peak_ok classical

present if lsb_score >= t_lsb
        or freq_score >= t_freq
        or class_score >= t_class

family_hint:
  lsb        if lsb_score is the unique max above its t
  frequency  if freq_score is the unique max above its t
  mixed      if both specialists fire
  classical  if only class_score fires
  unknown    if present but no specialist
```

Cutoffs come from M1 operating_point, not hand-tuned 0.48.

**Calibration.** After M1, optionally set

```yaml
calibration_path: configs/calibration.fitted.json
```

Refit on **bench-holdout** data, never on the frozen bench seeds. Pytest
forces `apply_calibration: false` or identity via a test config.

**JSON schema_version 2** adds `family_hint`, `lsb_score`, `freq_score`,
`operating_point_id`.

**Success:** on the frozen bench, specialist-OR FPR <= legacy FPR and LSB+DCT
TPR both >= legacy. CLI Top lists specialists first (already started).

### M4 -- Latent, gated, no downloads

- `tree_ring_inversion`: if `diffusers` **and** a local SD checkpoint path
  in config exist, run a **short** DDIM inversion on 64px center crop and
  score latent FFT rings. Else skip (current behavior).
- Do **not** pip-install 4GB weights in this upgrade.
- Gaussian Shading stays skip.
- WMD prune: keep dataset CLI; add a pytest with n=4, 1 round (already
  have tiny). Do not call it from single-image `scan`.

**Success:** skip path still tested; inversion path tested only if a fixture
flag `VEILSCAN_SD_CKPT` is set (CI never has it).

### M5 -- Production polish (only after M1)

- README truth (deep heads on, complementary split, how to cook).
- `veilscan doctor`: torch/cuda, ckpt hashes vs manifest, calibration path,
  bench protocol age.
- Tile path: confirm megapixel scan still uses specialist OR.
- ONNX of ResidualCNN already exists; add FSNet ONNX only if FFT export
  works (likely not). Do not block on it.
- One `docs/RESULTS_v0.4.md` that *replaces* the 0.2.1-0.2.4 stack as the
  number people read.

---

## 5. Explicit non-goals (this upgrade)

- Watermark removal / purification.
- SynthID private verifier.
- Mandatory DINOv2 / CLIP / HF fetch.
- UniFreq-100K in-tree (license + size). Optional later as an *external*
  bench adapter, not a git blob.
- Training one net on all families again (we already proved that fails).
- Gradio from a Grok Build command.
- Relicensing; repo stays private proprietary.

---

## 6. Priority backlog

**P0 (this upgrade's spine):**

1. Frozen `bench` protocol + `operating_point.json`.
2. README / LIMITATIONS match v0.2.4 reality.
3. Specialist-OR fusion behind a flag, default on once bench says so.
4. `family_hint` in JSON.

**P1:**

5. Vectorize DCT/spread embedders + cover cache.
6. Patchwork: measure, then specialist or unsupported.
7. Deep LOAO folders (holdout lsb / holdout dct).
8. Calibration path enabled with a test config, not by overwriting identity.

**P2:**

9. Optional inversion if local SD exists.
10. Social-JPEG recipe (WhatsApp-ish Q).
11. Heatmap IoU vs generator residual for DCT/DWT.
12. `veilscan doctor`.

**Won't:** stripper, SynthID clone, UniFreq in git, one-net-to-rule-them-all.

---

## 7. Risks

| Risk | Mitigation |
|------|------------|
| Bench overfits our generators | Frozen seeds; photo vs sine split; later external adapter |
| Specialist-OR raises FPR | Compare to legacy on the same JSON; flag `fusion.mode` |
| Enabling fitted cal breaks pytest | Tests pass an identity config |
| FSNet train with LSB again | Skill + train_lite help text + RESULTS warning; add a unit test that `--families` listing `lsb` prints a hard warning |
| Dual-use | No remover. Generators stay eval-only. |
| Stale README | M1 includes the rewrite |

---

## 8. Acceptance (v0.4.0 ship)

All of these, or we do not call it 0.3.0:

- [ ] `veilscan bench` produces `docs/bench/latest.json` from
      `configs/bench_protocol.yaml` (n>=50 in the real run; pytest n=2).
- [ ] `configs/operating_point.json` cites that bench. Engine uses it.
- [ ] README no longer says deep plugins are off. Complementary split is
      documented.
- [ ] JSON has `family_hint` + `schema_version` 2.
- [ ] Frozen-bench identity: photo-cover FPR <= 0.05 at the operating
      threshold; LSB and DCT TPR >= 0.80.
- [ ] JPEG70 frequency TPR retention >= 0.70 or LIMITATIONS explains the miss
      with the number.
- [ ] Patchwork is either TPR>=0.6 @ FPR 0.05 or explicitly unsupported
      (weight 0).
- [ ] FSNet train path warns if `lsb` is in `--families`.
- [ ] pytest green; secret scan; private push.
- [ ] Still no remover.

**v0.4.0 is an operating-point release, not a new-architecture release.** v0.3.0 already shipped keyless decode.

---

## 9. Suggested execution order (when implementing)

Week 1: M1 bench + README truth (even if n=20 first, then scale to 50).  
Week 2: M3 specialist-OR + schema 2, A/B vs legacy on the same JSON.  
Week 3: M2 vectorize embeds, patchwork decision, LOAO ckpts.  
Week 4: M4/M5 only if week 1-3 numbers hold.

If time is one session: **do M1 at n=20 plus README**, not a new net.

---

## 10. Citations still in force

Ao et al. AWPD/FSNet arXiv:2603.06723 (LSB/Patchwork vs frequency nets).  
Pan et al. WMD ECCV 2024 / arXiv:2403.15955.  
Fridrich RS 2001; Westfeld-Pfitzmann chi-square; Wen et al. Tree-Ring 2023.  
See `docs/RESEARCH.md`.
