# VeilScan (folder rules)

Agnostic invisible-watermark presence detector plus keyless plaintext decode. Local ensemble.

- Package: `src/veilscan`
- Docs: `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/ETHICS.md`
- Do not add watermark removal.
- Generators and `embed-text` are eval-only.
- Deep detectors must skip when no checkpoint exists.
- Do not run Gradio/uvicorn as a never-exit child of an agent job.
- ASCII punctuation in public files. UTF-8 no BOM.
- Tests: `py -3 -m pytest` from this folder after `pip install -e .`
- CLI: `py -3 -m veilscan scan PATH` and `py -3 -m veilscan decode PATH`
- Next science contract: `docs/NEXT_MASSIVE_UPGRADE.md`. v1.8.0 Q-table DCT inspect. BSDS camera sidecar stays v1.4 128px. Generator present stays ~0.67. Do not promote 256px OP. No from-scratch Q50. No remover. Do not train FSNet on `data/covers/camera`. JPEG50 DCT leftover stays 0.46 until a probe beats 0.40 FSNet-head TPR@5%.
- Fusion `peak_ok` lives in `configs/default.yaml`, not hardcoded.
