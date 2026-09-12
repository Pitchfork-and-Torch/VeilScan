# VeilScan upgrade plan (v0.1.0 -> production-grade AWPD)

**Status:** historical v0.1 -> v0.2 map. Written 2026-08-20.
**Next science contract:** `docs/NEXT_MASSIVE_UPGRADE.md`. Current package v2.0.0 (hunt Phase 0+1). Presence stack still v1.8 (`docs/RESULTS_v1.8.md`). JPEG50 leftover parked.
Phases 2-4 sprint: `docs/PHASES_2_3_4.md` (landed).  
**Identity:** presence plus keyless plaintext decode. No removal.

This document is the next-massive-upgrade contract. Phase 0 items listed in
section 8 are intended to land in the same working tree as this file.

---

## 1. Executive summary

VeilScan v0.1.0 is a real local AWPD prototype: a plugin ensemble with a
stable `DetectionResult` contract, a complementary classical+frequency suite,
gated deep nets, optional WMD, tiling, heatmaps, synthetic self-test, and
honest docs. It already encodes the most important 2026 research lesson from
AWPD/FSNet (Ao et al., arXiv:2603.06723): **frequency-centric nets fail on LSB
and Patchwork**, so bit-plane tests must stay in the fusion core.

It is not yet a reference implementation. Scores are sigmoid-mapped heuristics,
not calibrated probabilities. The self-test is small-n synthetic sine covers,
not UniFreq-100K leave-one-algorithm-out with attacks. Deep models have
architectures (FSNet-lite ASPM+DMSA, ResidualCNN) but no shipped checkpoints.
WMD is a shortened 8-step single-image offset, not iterative pruning on a
dataset. Tree-Ring is a pixel-FFT ring proxy, not DDIM inversion. CUDA is
detected but the default torch install at authoring was CPU-only.

**North star:** a local AWPD tool a forensic analyst can trust at low FPR,
extend by dropping a detector file, and train on one consumer GPU, without
ever growing a remover or a closed-verifier clone.

**Highest-leverage next moves (in order):**

1. Make evaluation honest: per-detector AUC tables, leave-one-family-out
   skeleton, attack retention, calibration curves. (Without this, every later
   architecture change is unmeasurable.)
2. Move fusion knobs (`peak_ok`, weights, threshold) fully into YAML and fit
   a simple per-head calibration map from synthetic+attack data.
3. Close LSB/Patchwork with *better classical math* (full Fridrich RS quadratic,
   Dumitrescu SPA, sequential chi-square) rather than hoping FSNet will save them.
4. Train ResidualCNN + FSNet-lite on attack-augmented multi-family data with
   true hold-one-family-out; only then let them vote.
5. Add *optional* gated inversion / strong residual proxies for Tree-Ring and
   Gaussian Shading. Never a SynthID decoder.

Target bars (from the upgrade brief, to be re-estimated after Phase 0 metrics):
leave-one-family-out mean AUC >= 0.90; TPR @ 1% FPR >= 0.70; >= 80% of clean
AUC retained under JPEG/resize/crop/noise/blur; lite checkpoints trainable on
one RTX-class GPU; classical path near real-time on CPU for megapixel images.

---

## 2. Current state diagnosis

Evidence is from the tree as of v0.1.0 (`2dcc70c` on private GitHub), local
pytest (19 passed), and measured selftests / CLI scans in-session.

### 2.1 What exists (do not invent)

| Area | Where | What it actually does |
|------|-------|------------------------|
| Contract | `src/veilscan/types.py` | `DetectionResult` (score, confidence, explanation, heatmap, extras, skipped, tier). `EnsembleResult`. `AnalyzeContext`. |
| Registry | `registry.py`, `detectors/__init__.py` | Lazy `register_all()` to avoid the import cycle (`base` must not load spatial). |
| Engine | `engine.py` | Sequential run, crash-to-skip, tiling with max score per detector and overlap-averaged heatmaps. |
| Fusion | `ensemble.py` | Skip if `skipped` or `confidence <= 0.05`. Weighted mean of all active heads, plus peak/top-3 of a **hardcoded** `peak_ok` set. Final score = 0.20 full_mean + 0.20 ok_mean + 0.20 top_mean + 0.40 peak. Threshold default **0.48**. Uncertainty = std of all active scores (including uncalibrated extras). |
| Spatial | `detectors/spatial.py` | Chi-square (RGB mean p-value, not gray). RS with mask `[0,1,1,0]`, *linear* payload hat (not Fridrich quadratic; no fully flipped second pass). SPA pair-family imbalance. Bit-plane RGB LSB entropy. Histogram even/odd gap. Patchwork adjacent-difference peak (explicitly weak, conf 0.35). |
| Frequency | `detectors/frequency.py` | 8x8 DCT mid-band (recalibrated log energy). DFT 1/f residual. Haar DWT. Hybrid DWT-DCT-SVD. Tree-Ring circular FFT (min angular CV + radial excess vs 1/f). Fourier-Mellin log-polar (low conf). Block multi-scale residual CV. |
| Residual | `detectors/residual.py` | KV SRM-lite + 9x9 co-occurrence entropy. YCbCr/HSV. Denoising reconstruction. Skew/kurtosis + row bispectrum proxy. |
| Deep | `models/fsnet.py`, `residual_cnn.py`, `detectors/deep.py` | FSNet-lite: grayscale ASPM DCT gate + 3-stride conv backbone + DMSA (K=8). ResidualCNN: 5x5 HP stem. Both **skip** without `checkpoints/*.pt`. Input forced to 128x128. |
| WMD | `detectors/blackbox.py`, `models/wmd_net.py` | Skip without `--reference-dir`. 8 Adam steps, softmax-on-clean + linear-on-suspect, **no iterative prune**, **no re-init**, expands one suspect to the clean batch size. |
| Foundation | `detectors/foundation.py` | Patch-PCA reconstruction on the *same* image. No DINOv2. |
| Generators | `generators/__init__.py` | Sine+blob synthetic covers. Families: lsb (rate 1.0), dct (amp 14), dwt (0.2), svd (0.04), patchwork, spread, tree_ring (multiplicative FFT ring, strength 0.4), hidden_approx. |
| Eval | `eval/harness.py`, `metrics.py`, `robustness.py` | `run_synthetic` (ensemble score only). `run_folder_eval`. `run_robustness` (one family, attacks listed). Metrics: Mann-Whitney AUC, TPR@FPR, F1. Attacks: identity, jpeg 90/70/50, resize_90, crop_90, noise, blur, jitter. **No LOAO. No per-detector table. No calibration curve.** |
| Train | `scripts/train_lite.py` | 80 steps default, batch 8, 128px, mixed families, no holdout, no attacks, no val split, CPU DataLoader `num_workers=0`. |
| CLI | `cli.py` | `scan`, `batch`, `embed` (eval), `eval-folders`, `selftest`, `robustness`, `list-detectors`. |
| Config | `configs/default.yaml` | weights, threshold 0.48, tile 1024. `jpeg_quality_probe: 95` is **loaded and unused**. `peak_ok` is **not** in YAML (hardcoded in `ensemble.py`). |
| Packaging | `pyproject.toml`, `Dockerfile`, `LICENSE` | Python >=3.11, proprietary license, Docker copies src+configs+docs, no CI workflow in-tree. |
| Tests | `tests/` | 19 tests: Haar roundtrip, generators shape, chi-square/RS ordering, DCT/DWT/tree-ring pairwise, ensemble LSB mean gap, deep/WMD skip, fuse peak not drowned, CLI list/version. SPA test does **not** require marked > cover. |

### 2.2 Measured behavior (this machine)

Synthetic selftest n=4, 128px (post-calibration fusion): overall AUC ~0.83,
TPR@5% FPR ~0.59, F1 ~0.72. DCT/DWT/spread/patchwork rank near 1.0 on that
draw. LSB rank ~0.94 with thresholded F1 lagging. Tree-Ring rank mixed, F1
often 0. SVD often at or below chance.

CLI local pair (192px sine cover vs matched embeds):

| Image | present | score |
|-------|---------|-------|
| cover | no | 0.424 |
| LSB | yes | 0.492 |
| DCT | yes | 0.661 |
| DWT | yes | 0.701 |
| spread | yes | 0.745 |
| tree_ring | no | 0.418 |

Cover still lights `block_multiscale` (~0.97) and `histogram` (~0.73). Those
heads are *excluded from* `peak_ok`, which is why the ensemble stays below
threshold. That is a fusion patch, not a calibrated cover model.

Deep skip is real: `fsnet_lite` / `residual_cnn` report missing checkpoints.
WMD skip is real without `--reference-dir`.

### 2.3 Strengths to keep

- Complementary failure-mode design is correct and documented in
  `docs/RESEARCH.md` section 3.
- Plugin skip contract is enforced in tests (`test_deep_skipped_without_ckpt`,
  `test_wmd_skipped_without_reference`).
- Independent reimplementation of ideas with citations (no private weights).
- Ethics file forbids a remover; generators are labeled eval-only in CLI help.
- Tiling + heatmap overlay already exist.

### 2.4 Gaps and technical debt

1. **Scores are not probabilities.** Each detector uses hand-tuned
   `score_from_stat` centers in `dsp.py`. Recalibration was done against sine
   covers, so natural photos will shift. No temperature, isotonic, or
   conformal layer. Threshold 0.48 is an operating point on that synthetic
   draw, not TPR@1% FPR.
2. **`peak_ok` is a magic set** in `ensemble.py` lines 37-45. Weights live in
   YAML; the more important allowlist does not. Drift risk.
3. **Uncertainty is the wrong statistic.** `std` of all active scores includes
   low-confidence extras (Fourier-Mellin, patchwork). Disagreement among
   `peak_ok` would be more honest.
4. **RS is incomplete.** `_rs_payload_estimate` is `|d0-d2|/(|d0|+|d2|)`.
   Fridrich 2001 needs the flipped-image second pass and a quadratic. Local
   scans showed RS barely moving on full-rate LSB (cover 0.38 vs LSB 0.39).
   Chi-square is carrying LSB.
5. **SPA test was relaxed** (`tests/test_spatial.py`) because neighbor-LSB
   correlation saturates on steep quantized ramps. The detector still votes
   with conf 0.80 and is *not* in `peak_ok`, but it still pollutes
   `full_mean` and CLI "Top" lists (sorted by score*weight, not peak_ok).
6. **WMD is not WMD.** 8 steps, no prune, trains from scratch per image,
   conf 0.55. Fine as a stub; dangerous if reported as ECCV-parity.
7. **FSNet-lite is not FSNet.** Grayscale stem (paper is RGB 2D-DCT gate),
   tiny backbone (not ResNet-50), no UniFreq training, 128px resize destroys
   high-frequency payload geometry.
8. **Training has no protocol.** No holdout family, no JPEG augment, no
   early stopping, default 80 steps. Cannot support the 0.90 LOAO bar.
9. **Eval harness does not match the research protocol.** `run_synthetic`
   only stores ensemble scores. You cannot see that `dct` moved +0.44 on
   spread while `chi_square` did not, without an ad-hoc dump. No LOAO.
   Robustness is JSON-only, one family, n=4.
10. **`jpeg_quality_probe` is dead config.** ELA / JPEG-ghost is a cheap
    missing head.
11. **Covers are not photos.** `synthetic_cover` is shaded sines + cubic
    blobs. Histogram/block_multiscale false-fire on this domain. Need
    photographic or high-res noise textures before claiming FPR.
12. **No CI.** Private GitHub repo has no workflow. Docker image does not
    copy `tests/` or `scripts/`.
13. **Heatmaps are un-scored.** No IoU / pointing-game even when the
    generator could provide a residual GT.
14. **CLI "Top" is misleading.** `_print_result` sorts all live detectors by
    raw score, so cover reports `block_multiscale=0.97` first even when the
    ensemble said no.

### 2.5 Research alignment

`docs/RESEARCH.md` already maps Westfeld, Fridrich RS, SRM/SRNet, HiDDeN,
StegaStamp, Stable Signature, Tree-Ring, WMD (arXiv:2403.15955), AWPD/FSNet
(arXiv:2603.06723), SynthID-image (arXiv:2510.09263). The code follows that
map at the *idea* level. The missing alignment is **protocol**: UniFreq-style
leave-one-algorithm-out, attack-augmented training, and calibrated TPR at low
FPR. That is the upgrade, not another random residual CNN.

---

## 3. Research gap analysis (2025-2026 AWPD)

| Frontier | Paper / cue | VeilScan now | High-leverage fit |
|----------|-------------|--------------|-------------------|
| AWPD task + UniFreq-100K | Ao et al. 2026 | Named in docs; dataset not bundled; no LOAO | Build a *local* LOAO harness on our generators first; optional later UniFreq subset if license allows. Do not scrape SynthID. |
| FSNet ASPM+DMSA | same | Lite grayscale reimplementation, untrained | Train on attack mix; RGB ASPM; do not pretend ResNet-50 parity until measured. |
| LSB/Patchwork vs nets | AWPD table acc < 0.60 | Classical heads exist; RS/SPA underpowered | Invest in RS quadratic + SPA, not a bigger ViT. |
| WMD offset + prune | Pan et al. ECCV 2024 | 8-step stub | Implement prune+reinit as a **dataset** CLI (`veilscan wmd-scan DIR --reference DIR`). Keep skip on single image. |
| Tree-Ring | Wen et al. 2023 | Pixel FFT rings | Optional gated DDIM inversion backend (diffusers), default off, no mandatory download. Until then, improve circular-consistency scoring and test inversion residuals only when the operator opts in. |
| Stable Signature | Fernandez et al. ICCV 2023 | Absorbed into "learned residual" | Fine-tune ResidualCNN on HiDDeN/StegaStamp *approximations* plus any public encoder we can vendor as optional extra. |
| Gaussian Shading | latent seed | Nothing specific | Honest miss unless inversion or a published pixel proxy appears. Do not fake a detector. |
| SynthID-image | Gowal et al. 2025 | Named, no verifier | Never reverse a private verifier. Treat as unknown multi-scale residual. |
| SRNet / YeNet | Boroumand 2018 | Tiny ResidualCNN | After data protocol exists, a real SRNet-scale model is Phase 1/2, still gated. |
| Foundation anomaly | DINOv2/CLIP bias (Yin Fourier) | Patch-PCA only | Optional offline weights if already on disk; never a required fetch. |
| C2PA | provenance manifests | Not in tree | Stretch: *correlate* C2PA presence as a side channel, never as a watermark decode. |
| Video | IW at scale | Images only | Stretch Phase 4+. |

**Philosophy filter:** if a method needs a secret decoder, a cloud API, or a
remover, it is out. If it is a presence cue that generalizes across unknown
embedders, it is in.

---

## 4. Phased roadmap

### Phase 0 / Quick wins (1-2 weeks) -- MEASURE AND STABILIZE

**Goals:** stop flying blind; make fusion configurable; CI green; docs match
code.

**Success metrics:**
- `veilscan selftest --per-detector` prints per-head AUC by family.
- `veilscan loao` runs (classical path = per-family isolation report; deep
  LOAO documented as requiring `--train-holdout`).
- `peak_ok` and fusion mix coefficients live in YAML. Tests fail if a
  detector is renamed without updating config.
- GitHub Actions on private repo: `pytest` on push.
- Cover CLI no longer leads with excluded extras.

**Work items:**
1. Per-detector + LOAO skeleton in `eval/harness.py`.
2. YAML `peak_ok`, `fusion` mix, `uncertainty_from: peak_ok`.
3. CLI Top list uses `peak_ok` first.
4. `.github/workflows/ci.yml`.
5. Point RESEARCH / ARCHITECTURE / AGENTS at this plan.
6. Delete or use `jpeg_quality_probe` (prefer a small JPEG-residual head).

**Files:** `ensemble.py`, `config.py`, `configs/default.yaml`, `eval/harness.py`,
`cli.py`, `engine.py`, `tests/test_eval.py`, `.github/workflows/ci.yml`,
`docs/*`.

**Tests:** per-detector table shape; LOAO keys; YAML peak_ok loaded; fuse still
passes `test_fuse_peak_not_drowned`; CI pytest.

**Effort / risk:** low. Risk is over-fitting YAML to sine covers. Mitigation:
treat Phase 0 numbers as *diagnostic*, not claimed SOTA.

### Phase 1 -- Detection power (2-4 weeks)

**Goals:** LSB/Patchwork actually move RS/SPA/chi-square on photographic
covers; residual features less domain-fragile; FSNet-lite RGB ASPM.

**Work items:**
- Fridrich RS with dual flip + quadratic p-hat; test against known
  embedding-rate grid (0, 0.05, 0.2, 0.5, 1.0).
- Dumitrescu SPA embedding-rate estimator; lower confidence until tests
  show marked > cover on photographic textures, not only sines.
- Sequential chi-square on prefix blocks (Westfeld original use).
- JPEG ELA / quality-probe residual detector using existing
  `image_io.jpeg_roundtrip`.
- Replace `synthetic_cover` with a two-source mix: sines (regression) +
  filtered natural-like noise / optional local photo folder (eval only).
- FSNet-lite: RGB ASPM, keep skip-without-ckpt.
- Hybrid spatial-deep: concatenate RS/chi extras as scalar side features
  into the CNN head (optional).

**Files:** `detectors/spatial.py`, `detectors/residual.py`, `detectors/frequency.py`,
`models/fsnet.py`, `generators/__init__.py`, new `detectors/jpeg_ela.py`,
tests.

**Success:** on n>=20 photographic-like covers, RS and chi-square each
achieve AUC >= 0.85 on LSB rate>=0.5; Patchwork family AUC >= 0.75 with
honest conf; cover mean ensemble < threshold - 0.05.

**Risk:** RS math bugs. Mitigation: unit tests on synthetic LSB rate ladders
with known p, not just "score went up."

### Phase 2 -- Training, data, robustness, calibration (3-6 weeks)

**Goals:** checkpoints that deserve their YAML weights; TPR at 1% FPR that
means something; JPEG retention.

**Work items:**
- Dataset builder: N covers x families x attack grid, on-disk or streaming.
- `train_lite.py` rewrite: val split, `--holdout-family`, JPEG/resize augment,
  early stop, seed logging, write `checkpoints/manifest.json`.
- Isotonic or temperature scaling **per detector** and for the ensemble,
  fit on a held-out synthetic+attack set, stored in `configs/calibration.json`.
- Conformal / quantile wrapper optional (target FPR 1% and 5%).
- Robustness harness: all families x DEFAULT_ATTACKS, report retention
  `auc_attack / auc_identity`.
- Heatmap quality if generator can export a residual mask (DCT/DWT/tree-ring
  can).

**Success:** LOAO mean AUC >= 0.85 on expanded synthetic+attacks as a
*checkpoint* toward 0.90; TPR@1% FPR reported not guessed; identity-AUC
retention >= 0.80 on jpeg_70 for DCT/spread (LSB may drop; report honestly).

**Risk:** training on CPU is too slow; CUDA wheel may be needed. Do not block
classical path on GPU.

### Phase 3 -- Latent and advanced cues (4-8 weeks, gated)

**Goals:** Tree-Ring / Stable Signature presence without claiming SynthID.

**Work items:**
- Optional `diffusers` inversion residual detector, `requires_extra=True`,
  skip if package or weights missing (same pattern as checkpoints).
- Multi-resolution tiling of FFT ring detector (not only 128px).
- Do **not** ship Gaussian Shading recovery of seeds.
- Dataset-hygiene mode: folder scan with prevalence prior (WMD full prune
  algorithm here, not in single-image `scan`).

**Success:** Tree-Ring approx + any inversion-opt-in path documented with
separate metrics. Default install still works with numpy/cv2/torch CPU.

**Risk:** inversion is slow and model-version brittle. Keep it optional forever.

### Phase 4 -- Engineering and production (ongoing)

**Goals:** megapixel CPU classical path; GPU batch; ONNX export of trained
nets; Docker that includes tests; stable JSON schema (`schema_version`);
Gradio remains operator-launched, never from Grok Build.

**Work items:**
- Vectorize remaining Python loops (RS groups already numpy; LSB heatmap
  8x8 nested Python in `_lsb_plane_heatmap` is a real cost).
- Batch `analyze_images`.
- `torch.jit` / ONNX for ResidualCNN and FSNet-lite after they have
  checkpoints.
- `schema_version` on `EnsembleResult.to_json()`.
- CI matrix: py3.11 / 3.13, CPU.
- Dockerfile: multi-stage, non-root, copy tests, `HEALTHCHECK` not a server.

**Stretch:** video keyframe loop; C2PA sidecar correlation; foundation
anomaly iff weights already on disk.

---

## 5. Detailed technical designs (highest impact)

### 5.1 Per-detector eval + LOAO skeleton

**Why:** complementary-failure-mode design cannot be verified from one
ensemble number. In-session dumps showed `dct` mid_energy 0.96 -> 1.94 on
DCT embeds while ensemble was dominated by other heads until `peak_ok`.

**Interface:**

```
run_synthetic(..., per_detector=True) -> {
  overall, families,
  detectors: {name: {family: {auc, mean_cover, mean_marked}}}
}

run_loao(n, size) -> {
  held_out: {family: {auc, note}},
  protocol: "classical heads are family-agnostic; deep LOAO needs train --holdout"
}
```

For untrained nets, LOAO of *training* is N/A. Report that explicitly.
Phase 2 fills `scripts/train_lite.py --holdout-family dct`.

**Eval protocol:** same Mann-Whitney AUC as `eval/metrics.py`. n default 8
for CLI, 3 for pytest.

### 5.2 Configurable fusion

**Why:** `peak_ok` hardcoded in `ensemble.py` is the actual decision surface.
YAML weights are secondary.

```yaml
fusion:
  mix: {full_mean: 0.20, ok_mean: 0.20, top_mean: 0.20, peak: 0.40}
  peak_ok: [chi_square, rs_analysis, bitplane, dct, dwt, hybrid_dds, tree_ring_spectral]
  uncertainty: peak_ok   # not all extras
  top_k: 3
```

`fuse(..., peak_ok, mix)` stays pure. Engine passes config.

**Expected effect:** fewer silent fusion regressions; CLI Top matches the
score that drove `present`.

### 5.3 Fridrich RS (Phase 1)

**Why:** local LSB scan, RS 0.381 -> 0.393. Chi-square did the work.

**Approach:** implement the published dual statistics (M and -M) on original
and LSB-flipped image; solve the quadratic for p; map p through a
*calibrated* curve. Unit test: embedding rate ladder, estimated p monotone.

**Weak families helped:** LSB, to a lesser extent histogram-equalized
payloads. Will not help Tree-Ring.

### 5.4 Attack-augmented training (Phase 2)

**Why:** AWPD paper and StegaStamp both treat JPEG as the default world.
`train_lite.py` never JPEG-compresses.

**Approach:** wrap `apply_attack` in the Dataset `__getitem__` with a
probability schedule. Hold out one generator family per run. Write
manifest: `{families, attacks, holdout, steps, auc_val}`.

**Gating:** detectors still skip if no file. Never load random weights.

### 5.5 Optional inversion (Phase 3)

**Why:** Tree-Ring lives in inverted latent FFT (Wen et al.). Pixel rings
are a proxy; local test tree_ring score 0.418 (below threshold).

**Approach:** `TreeRingInversionDetector` with `requires_extra=True`. If
`diffusers` and a local SD checkpoint exist, invert and score ring
consistency in latent FFT. Else skip with a clear explanation.

**Non-goal:** removing the ring.

### 5.6 Calibration layer (Phase 2, schema in Phase 0)

```json
{
  "schema_version": 1,
  "method": "affine_logit",
  "detectors": {"chi_square": {"a": 1.0, "b": 0.0}, "...": {}},
  "ensemble": {"a": 1.0, "b": 0.0, "threshold": 0.48},
  "fit": {"n": 0, "protocol": "unfitted"}
}
```

Phase 0 ships the schema and identity maps. Phase 2 fits them.

---

## 6. Priority backlog

**P0 (do now):**
- Per-detector selftest + LOAO skeleton + tests
- YAML `peak_ok` / fusion mix; uncertainty from peak_ok
- CLI Top uses peak_ok
- CI pytest on private GitHub
- This plan linked from README / ARCHITECTURE / AGENTS
- Use or remove `jpeg_quality_probe`

**P1:**
- Full RS + SPA + sequential chi-square with rate-ladder tests
- JPEG ELA detector
- Better covers (not only sines)
- Train script holdout + JPEG augment
- Calibration.json identity schema then fit

**P2:**
- RGB ASPM, real SRNet-scale option
- WMD dataset prune CLI
- Optional inversion
- ONNX, batch API, heatmap IoU
- Photographic FPR study

**Won't do:**
- Watermark remover / purifier / "clean this PNG"
- SynthID private verifier clone
- Mandatory DINOv2/HF download
- Gradio launched from Grok Build
- Claiming copyright or generator ID

---

## 7. Risks, ethics, mitigations

| Risk | Mitigation |
|------|------------|
| Dual use (detection aids stripping) | Keep ETHICS.md; no remover; generators stay `embed` eval CLI |
| Over-claiming WMD/FSNet parity | Name "lite" / "stub"; skip without ckpt; cite papers |
| Sine-cover overfit | Phase 0 numbers are diagnostic; Phase 1 adds other covers |
| CUDA/torch CPU split | Classical path never requires GPU |
| Closed watermarks | Honest LIMITATIONS; no fake SynthID head |
| Score used as legal evidence | LIMITATIONS + CLI explanation text |
| Import cycle regression | Keep lazy `register_all`; test collection imports `cli` |
| Private GitHub leak | Proprietary LICENSE; secret scan before every push |

Non-goals reaffirmed: no decode product, no removal, no cloud requirement,
no random-weight voting.

---

## 8. Immediate next actions (1-2 week sprint)

Begin in this tree, this session where possible.

| # | Task | Files | Tests | Done when |
|---|------|-------|-------|-----------|
| 1 | Land this document | `docs/UPGRADE_PLAN.md` | n/a | Linked from README + ARCHITECTURE + AGENTS |
| 2 | Per-detector metrics in harness | `eval/harness.py`, `cli.py` | `tests/test_eval.py` | `selftest --per-detector` prints a head x family AUC table |
| 3 | LOAO skeleton | `eval/harness.py` `run_loao` | same | `veilscan loao --n 3` exits 0 with per-family AUC |
| 4 | Configurable fusion | `config.py`, `default.yaml`, `ensemble.py`, `engine.py` | update `test_fuse_*` | changing YAML peak_ok changes fusion without code edit |
| 5 | CLI Top = peak_ok first | `cli.py` | CLI test optional | cover scan does not lead with block_multiscale when excluded |
| 6 | CI | `.github/workflows/ci.yml` | workflow pytest | push runs tests |
| 7 | JPEG residual head (if time) | `detectors/jpeg_ela.py` | pairwise jpeg-robust family | uses `jpeg_quality_probe`; skip-free classical |

**Phase 0 checkpoint (accept):** pytest green including new eval tests; plan
on `main`; selftest still ranks DCT/DWT/spread above cover; deep/WMD still
skip without extras.

**Explicitly not in this sprint:** training 80+ GPU epochs, inversion,
UniFreq download, ONNX, remover.

---

## Appendix A. Detector roster (v0.1.0)

`chi_square`, `rs_analysis`, `sample_pairs`, `bitplane`, `histogram`,
`patchwork`, `dct`, `dft`, `dwt`, `hybrid_dds`, `tree_ring_spectral`,
`fourier_mellin`, `block_multiscale`, `srm`, `color_spaces`,
`reconstruction`, `higher_order`, `residual_cnn` (gated), `fsnet_lite`
(gated), `wmd` (gated), `foundation`.

`peak_ok` today: chi_square, rs_analysis, bitplane, dct, dwt, hybrid_dds,
tree_ring_spectral.

## Appendix B. Citation keys

See `docs/RESEARCH.md` section 6. Do not drop citations when rewriting heads.
