# VeilScan (folder rules)

Agnostic invisible-watermark presence detector plus keyless plaintext decode. Local ensemble.

- Package: `src/veilscan`
- Docs: `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/ETHICS.md`
- Do not add watermark removal.
- Generators and `embed-text` are eval-only.
- Deep detectors must skip when no checkpoint exists.
- Do not run Gradio/uvicorn from a Grok Build command (never-exit / Job Object).
- ASCII punctuation in public files. UTF-8 no BOM.
- Tests: `py -3 -m pytest` from this folder after `pip install -e .`
- CLI: `py -3 -m veilscan scan PATH` and `py -3 -m veilscan decode PATH`
- Next science contract: `docs/NEXT_MASSIVE_UPGRADE.md`. v1.0.0 production cut. Camera sidecar ~0.758. Generator present stays ~0.67. Patchwork weight 0. JPEG-aware blend on jpeg_like arrays. No remover. No personal photos in git. Do not train FSNet on `data/covers/camera`.
- Fusion `peak_ok` lives in `configs/default.yaml`, not hardcoded.
