# Limitations (honest)

## Detection

- **LSB and Patchwork** vs modern nets: AWPD/FSNet leave-one-out accuracy was
  below 0.60. VeilScan therefore keeps classical bit-plane tests. Those tests
  themselves fail on latent-only generative marks.
- **Tree-Ring / Gaussian Shading / SynthID** in pixel space without inversion
  or a vendor verifier is weak. The circular-FFT detector is an approximation
  of Tree-Ring, not a replacement for DDIM inversion.
- **Single-image WMD** is undefined. WMD needs a clean reference set.
- **Untrained CNNs** skip until `checkpoints/*.pt` exist. This tree currently
  ships ResidualCNN + FSNet-lite checkpoints. They are complementary: Residual
  for LSB, FSNet for frequency. Do not train FSNet on LSB.
- **Calibration** against UniFreq-100K is not done (dataset not bundled).
  Generator OP: `configs/operating_point.json`. Camera sidecar:
  `configs/operating_point.camera.json` (BSDS500 test-64, n=50). n<50 is
  `provisional`. Do not quote either as ImageNet FPR.
- **Decision threshold** for default `present` is still the generator lock
  ~0.67 (`op-v0.4.0-locked-n50`). Camera stills sit near ensemble 0.71, so
  that lock false-fires on BSDS500 and DIV2K. The camera sidecar cut is ~0.758
  (`op-v1.0.0-camera-locked-n50`, FPR 0.05). Scan JSON adds `camera` when
  the sidecar exists. Default `present` is not flipped in v1.0. DIV2K
  valid-HR FPR at 0.67 is 0.72. UniFreq-100K is not in tree. JPEG70 DCT
  ensemble TPR@5%FPR is 0.64 (v0.7 was 0.44). JPEG-like arrays blend FSNet
  into the mix.
- Fusion mode stays `legacy`. specialist-OR FPR was 0.14 vs 0.05 in-sample.
  Nested even/odd holdout is reported under `ab.nested_holdout`; do not flip
  OR unless that nested FPR also holds.
- **JPEG70 LSB** dies on camera stills (TPR@5%FPR 0.06 on the v1.0 BSDS
  slice). Frequency DCT on that slice is 0.64 at JPEG70 vs 0.84 identity.
  Quote `docs/RESULTS_v1.0.md`, not older generator-photo DCT numbers.
- **Patchwork is unsupported** on camera stills. A keyless permutation-null
  pair-mean test (v0.9) is AUC 0.50 / TPR@5%FPR 0.08 on BSDS n=24. Camera
  ensemble TPR stays ~0.02-0.10. YAML weight is 0 so the head cannot pollute
  `full_mean`. A keyed specialist would be a new model, not this head.

## Robustness

JPEG, downscale, and diffusion regeneration destroy or hide many spatial marks.
A high score on a pristine PNG can collapse after social-media re-encode.
Spatial LSB usually dies at JPEG Q70 / Telegram recompress; frequency marks
may retain. The bench prints both numbers. Patchwork is measured; if TPR stays
weak it is labeled unsupported rather than given a fake YAML weight.

## Compute

Classical+frequency path: CPU, seconds per megapixel.
FSNet-lite / residual CNN training: needs CUDA PyTorch for comfort. The
machine has an RTX 4080; the default torch install at authoring was CPU-only.

## Decode

v0.5 recovers **keyless plaintext** and **localizes** LSB patches:

- sequential LSB in the full raster and in blindly found tiles
- PNG `tEXt` / `zTXt` / `iTXt`
- JPEG COM and common EXIF comment tags
- LSB-plane QR (OpenCV)
- optional JSteg (quantized DCT AC LSBs) when `jpeglib` is installed

A small LSB rectangle is sequential only inside that box. Short headers often
fit on one scanline (RGB interleaved). v0.3.2 scans each row for NUL-framed
tokens like `INV_WM:LEFT_EYE:2026` without coordinates. Tile autocorrelation
still localizes longer patches.

It will not print encrypted stego (Steghide / OpenStego with a password),
SynthID, Digimarc, Tree-Ring payloads, or HiDDeN-class neural marks.
Spatial LSB usually dies after JPEG / Telegram recompress. Hotspots can
still mark the patched region. A recovered string is not proof of authorship.

## Legal

A VeilScan score is not a copyright determination and not evidence of a
specific generator.
