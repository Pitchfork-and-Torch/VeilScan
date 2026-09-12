# VeilScan (folder rules)

Local watermark presence detector plus keyless decode plus forensic hunt extract.

- Package: `src/veilscan`
- Docs: `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/ETHICS.md`
- Do not add watermark removal.
- Generators, `embed-text`, and `gym` plants are eval-only.
- Deep detectors must skip when no checkpoint exists.
- Do not run Gradio/uvicorn as a never-exit child of an agent job.
- ASCII punctuation in public files. UTF-8 no BOM.
- Tests: `py -3 -m pytest` from this folder after `pip install -e .`
- CLI: `py -3 -m veilscan scan PATH`, `decode PATH`, `hunt PATH`, `gym`
- Science leftover (parked): JPEG50 DCT TPR 0.46. Do not recook FSNet. Do not
  flip `present`. Do not replace 128px BSDS lock. No from-scratch Q50.
- Hunt contract: `docs/NEXT_MASSIVE_UPGRADE.md`. v2.2.0 is Phase 0-3
  (carve + zsteg-class + JSteg + optional PATH adapters). Gym required TPR 1.0.
- Fusion `peak_ok` lives in `configs/default.yaml`, not hardcoded.
- Stay private GitHub, proprietary LICENSE. Do not document extractors on the lamp.
