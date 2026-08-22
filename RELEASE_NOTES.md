# v0.5.1

Doctor, camera-cover fetch, optional scan-JSON camera sidecar.

- `veilscan doctor`: torch/cuda, checkpoint sha256 vs `checkpoints/manifest.json`, generator OP required, camera OP optional
- `scripts/fetch_camera_covers.py` + `configs/camera_covers.manifest.json` (BSDS500 test-64, extracted JPEGs gitignored)
- Scan JSON may include `camera` when `configs/operating_point.camera.json` exists; omitted otherwise
- Generator lock unchanged (`op-v0.4.0-locked-n50`). Camera FPR not measured until n>=50 bench
- `--covers --write-operating-point` writes the camera sidecar only; refuses to overwrite the generator lock

Proprietary. Private repo.

# v0.5.0

Local presence detector plus keyless plaintext reader.

- Locked generator-photo operating point (threshold ~0.67, FPR 0.05 on that slice)
- Scan JSON `family_hint` (schema 2)
- `veilscan inspect` (scan + decode + HUD)
- Scanline token hunt in CLI and on the lamp
- Camera folder bench: `veilscan bench --covers DIR`
- Vectorized DCT / spread embeds
- FSNet `--families lsb` exits 2
- No stripper. Not a SynthID decoder.

Proprietary. Private repo.
