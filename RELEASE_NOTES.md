# v0.8.0

Second camera corpus (DIV2K valid-HR). Default `present` still not flipped.

- Fetch: `configs/camera_div2k_covers.manifest.json` (sha256 pinned). Extracted PNGs gitignored
- `--corpus-id` so a second pack cannot overwrite the BSDS camera OP
- DIV2K n=50 lock `op-v0.8.0-camera-div2k-locked-n50` threshold ~0.765
- FPR at generator 0.67 on DIV2K: 0.83. FPR at BSDS 0.753: 0.12
- Scan JSON `camera` sidecar stays BSDS. Fusion stays `legacy`
- Bench prints `fpr_at_locks` against existing generator and BSDS cuts

Proprietary. Private repo.

# v0.7.0

FSNet-lite recook on BSDS500 train stills. Frozen test-64 stays the camera bench.

- Cook: `--covers data/covers/camera-train --cover-mix 0.4 --families dct,spread,dwt,tree_ring` (no lsb). Refuses the frozen test pack
- Camera sidecar `op-v0.7.0-camera-locked-n50` threshold ~0.753 (was ~0.817). `t_freq` 0.875 (was 1.0)
- Camera identity DCT TPR@5%FPR 0.96 (was 0.06). DWT 0.98. Spread 1.00. Tree-ring 0.90
- Generator lock unchanged (~0.67). Default `present` not flipped. Fusion stays `legacy`
- ResidualCNN unchanged. Patchwork on camera stills still unsupported
- `veilscan doctor` hash for FSNet is `6e45dbba4852c612679f3c1f957fc8040f0077c4826e0dfab9a9002b9472e580`
- Probe helper: `scripts/probe_fsnet.py`

Proprietary. Private repo.

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
