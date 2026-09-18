# Phases 2-4 sprint contract (execute now)

Companion to `docs/UPGRADE_PLAN.md`. Identity unchanged: presence only, no decode, no remover.

## Phase 2 -- measure and calibrate

| Item | Done when |
|------|-----------|
| Affine-logit calibration schema + identity file | `configs/calibration.json` loads; a=1,b=0 is a no-op |
| `veilscan calibrate` fits a,b per head on synthetic covers/marks | Writes JSON; tests fit to tmp, not the identity file |
| Engine applies calibration before fusion | Existing unit tests that call detectors directly stay raw |
| Robustness across attacks, optional all families | `run_robustness` returns identity AUC and retention |
| Train holdout/JPEG | Already in `scripts/train_lite.py` (Phase 1). No full GPU train this sprint |

**v0.2.1 cook:** CPU ResidualCNN 50 steps (val_auc 0.74, live pair means still flat). Fitted cal JSON shipped but identity remains default. See `docs/RESULTS_v0.2.1.md`.

**Still later:** UniFreq download, CUDA cook, claiming TPR@1% FPR on photos.

## Phase 3 -- latent / dataset cues (gated)

| Item | Done when |
|------|-----------|
| Multi-scale pixel Tree-Ring | extras include `scales` scores; max fused |
| `tree_ring_inversion` | Skip unless `diffusers` + local SD path (never required) |
| WMD prune CLI | `veilscan wmd-scan DIR --reference-dir DIR` iterative keep-top; skip on single `scan` without refs |
| Gaussian Shading | Honest skip only (no seed recovery) |

## Phase 4 -- production edges

| Item | Done when |
|------|-----------|
| `schema_version` in JSON reports | Field = 1 |
| `analyze_images` batch API | List in, list out |
| ONNX export of ResidualCNN | `scripts/export_onnx.py` / `veilscan export-onnx` |
| Docker copies tests+scripts | Image still CLI, not a never-exit server |
| Version 0.2.0 | pyproject + package |

## Non-goals (still)

No watermark stripper. No SynthID verifier. No mandatory HF/DINOv2 fetch. No Gradio from Grok Build.
