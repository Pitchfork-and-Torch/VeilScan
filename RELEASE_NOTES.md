# v2.1.0

zsteg-class hunt. Presence OP unchanged.

- Native channel/bit/order LSB walks (no Ruby zsteg)
- Palette-index LSB before RGB expand
- Bitplane sheet + QR on bit planes (`--out` writes bitplanes.png)
- `--deep` uses bits 0-7
- Gym 8/8: prior four plus zsteg-rgb-bit0, alpha-lsb, palette-lsb, bitplane-qr

# v2.0.0

Forensic hunt extract. Presence OP unchanged.

Private GitHub only. Lamp stays generic.

- `veilscan hunt FILE` carves trailing/embedded files, dumps PNG/JPEG/GIF/WEBP container text, harvests FLAG{}
- `veilscan gym` plants four eval fixtures (tEXt, JPEG COM, trailing zip, unknown PNG chunk)
- Decode token rail rejects short photo LSB TAG:VAL (PTQ:B7C). INV_WM:LEFT_EYE:2026 still scores
- JPEG50 leftover remains parked. No FSNet recook. No remover

# v1.8.0

Q-table DCT inspect. No production recook. JPEG50 DCT TPR leftover stays 0.46.

Lamp: https://veilscan.jonbailey.xyz/

- `inspect_jpeg` / scan JSON / CLI print luma DQT steps at bins (3,4) and (4,3)
- DCT extras include `pair_34_43`. Score mix, present, and 128px lock unchanged
- Eval DCT amp 14 sits under Q50 luma steps ~51/56. No from-scratch Q50. No FSNet recook

# v1.7.0

Native 128 windows on large stills. No production recook.

Lamp: https://veilscan.jonbailey.xyz/

- ResidualCNN and FSNet score four corners plus center at native 128, then mean. Exact 128 is one window.
- BSDS 256px identity DCT TPR 0.06 -> 1.00. JPEG70 DCT 0.08 -> 0.90. FPR at 0.67 stays 0.02.
- Do not replace the 128px camera sidecar. Generator present stays ~0.67.
- Stage D Q50 recooks from v1.3 rejected (FSNet-head jpeg_50 DCT 0.40 -> 0.30 / 0.34). 128px JPEG50 DCT ensemble TPR stays 0.46.

# v1.6.0

Confirmation. No production recook. 128px BSDS camera sidecar unchanged.

Lamp: https://veilscan.jonbailey.xyz/

- DIV2K PNG sidecar `op-v1.6.0-camera-div2k-locked-n50` ~0.796. Generator FPR 0.59 (identity PNG, no JPEG blend)
- BSDS 256px TPR collapsed. Do not replace 128px lock. FPR 0.02 at 0.67 is not a win
- LOAO holdout-dct: production DCT TPR 0.98 -> 0.08. Specialist is still real
- `--policy` and schema 5 blend unchanged. Default present stays generator

# v1.5.0

Camera policy CLI. Default present stays generator.

Lamp: https://veilscan.jonbailey.xyz/

- `veilscan scan|inspect|batch --policy generator|camera|both`
- JSON `policy`. Camera sidecar still attached. Default generator even on JPEG files.
- Quality-aware blend from v1.4 unchanged. No recook.

# v1.4.0

Quality-aware FSNet blend. Schema 5. No recook.

Lamp: https://veilscan.jonbailey.xyz/

- `jpeg_freq_weight` from Q-table: 0.25 at Q90+, ~0.48 at Q70, 0.70 at Q50
- JPEG50 DCT TPR holds 0.46. Identity LSB 0.06 -> 0.14 (lighter high-Q blend)
- Camera sidecar `op-v1.4.0-camera-locked-n50` ~0.756. Generator lock ~0.67 FPR 0.08
- Fusion `legacy`. Patchwork weight 0. No stripper.

# v1.3.0

DCT-heavy Q50 FSNet fine-tune. Lamp lists inspect, JPEG container fields, and honest Q50 DCT limits.

Lamp: https://veilscan.jonbailey.xyz/

- Fine-tuned v1.2 weights with `--families dct,dct,dct,dwt,spread,tree_ring`
- FSNet-head jpeg_50 DCT TPR 0.26 -> 0.40. Ensemble 0.32 -> 0.46
- Identity DCT and JPEG70 DWT held. Generator lock ~0.67 FPR on BSDS 0.08
- Camera sidecar `op-v1.3.0-camera-locked-n50` ~0.758
- Fusion `legacy`. Patchwork weight 0. ResidualCNN unchanged. No stripper.

# v1.2.0

Q50-aware FSNet fine-tune. Native JPEG path from v1.1 stays.

Lamp: https://veilscan.jonbailey.xyz/

- Fine-tuned v1.0 FSNet on camera-train with `--jpeg-attacks jpeg_50,jpeg_50,jpeg_70,jpeg_90`
- From-scratch Q50 cook rejected (identity DCT TPR 0.96 -> 0.38)
- JPEG50 ensemble TPR@5%FPR: DCT 0.18 -> 0.32, DWT 0.42 -> 0.56, spread 0.30 -> 0.48, tree-ring 0.80 -> 0.94
- JPEG70 DWT 0.80 -> 0.88. Identity DCT held
- Camera sidecar `op-v1.2.0-camera-locked-n50` ~0.484. Generator lock ~0.67 FPR on BSDS 0.01
- Fusion `legacy`. Patchwork weight 0. ResidualCNN unchanged. No stripper. Proprietary.

# v1.1.0

Native JPEG path. File bytes decide `jpeg_like`, not only 8x8 ringing.

Lamp: https://veilscan.jonbailey.xyz/

- `jpeg_container` / `jpeg_quality_est` / `jpeg_subsampling` (schema 4)
- `jpeg_like` = JPEG SOI **or** blockiness >= 1.10
- Camera sidecar `op-v1.1.0-camera-locked-n50` ~0.739
- Generator 0.67 FPR on BSDS camera stills 0.66 -> **0.07** (lock not flipped)
- JPEG70 DWT TPR@5%FPR 0.30 -> **0.80** without recooking FSNet
- JPEG50 measured (DCT TPR 0.18). Not in the OP lock slice
- Identity LSB on JPEG rasters is no longer a field claim (PNG LSB unchanged)
- Fusion `legacy`. Patchwork weight 0. No stripper. Proprietary. Private repo.

# v1.0.0

First production cut. Presence ensemble plus keyless plaintext decode.

Lamp: https://veilscan.jonbailey.xyz/

- JPEG-hardened FSNet (`jpeg_prob` 0.7, camera-train mix 0.35, no lsb)
- Honest JPEG70 camera DCT: FSNet TPR@5%FPR 0.75 (was 0.58). Ensemble 0.64 (was 0.44)
- JPEG-aware blend when `jpeg_like` (schema 3: `jpeg_like`, `jpeg_blockiness`)
- Camera sidecar `op-v1.0.0-camera-locked-n50` ~0.758. Generator lock ~0.67 unchanged
- LOAO holdout-dct: FSNet DCT TPR 0.94 -> 0.06. Specialist is real
- Patchwork still unsupported (weight 0). Fusion stays `legacy`
- Probe attacks covers as well as marked (v0.7 JPEG70 leak fixed)
- No stripper. Proprietary. Private repo.

# v0.9.0

Patchwork unsupported on camera stills.

- Keyless permutation-null pair-mean test (public pairing, not the secret key)
- Head AUC 0.50 / TPR@5%FPR 0.08 on BSDS n=24. YAML weight 0
- Generator lock, BSDS camera sidecar, and DIV2K confirmation unchanged
- Fusion stays `legacy`. No stripper

Proprietary. Private repo.

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
