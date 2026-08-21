# VeilScan (folder rules)

Agnostic invisible-watermark **presence** detector. Local ensemble.

- Package: `src/veilscan`
- Docs: `docs/RESEARCH.md`, `docs/ARCHITECTURE.md`, `docs/ETHICS.md`
- Do not add watermark removal.
- Generators are eval-only.
- Deep detectors must skip when no checkpoint exists.
- Do not run Gradio/uvicorn from a Grok Build command (never-exit / Job Object).
- ASCII punctuation in public files. UTF-8 no BOM.
- Tests: `py -3 -m pytest` from this folder after `pip install -e .`
- CLI: `py -3 -m veilscan scan PATH`
- Upgrade contract: `docs/UPGRADE_PLAN.md` and `docs/PHASES_2_3_4.md`. Presence only. No remover.
- Fusion `peak_ok` lives in `configs/default.yaml`, not hardcoded.
