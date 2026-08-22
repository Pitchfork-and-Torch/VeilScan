# v0.6.0

Camera stills operating point. Default `present` still uses the generator lock.

- Sidecar `op-v0.6.0-camera-locked-n50` threshold ~0.817, FPR 0.05, n=50 BSDS500 test
- Generator lock unchanged (~0.67). Camera cover mean ~0.80, so 0.67 false-fires there
- Scan JSON `camera` sidecar; omitted only if the file is missing
- Fusion stays `legacy` (nested OR FPR 0.98 on camera). FSNet saturates on this pack
- LSB identity still separates (TPR@5%FPR 1.0). JPEG70 LSB and camera DCT do not
- `veilscan doctor` now reports the camera OP
- Extracted JPEGs stay gitignored. Archive sha256 pinned in the fetch manifest

Proprietary. Private repo.

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
