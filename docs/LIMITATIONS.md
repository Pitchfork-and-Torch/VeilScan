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
- **Calibration** of scores to camera-photo false-positive rates is not done
  against UniFreq-100K (dataset not bundled). `veilscan bench` measures a
  **generator** operating point (`configs/operating_point.json`). n<50 is
  `provisional`. Do not quote it as ImageNet FPR.
- **Decision threshold** is now `operating_point.threshold` ~0.67
  (`op-v0.4.0-locked-n50`, generator-photo identity+jpeg_70, n=50, FPR 0.05).
  Do not quote that as camera FPR. Fusion mode stays `legacy`: specialist-OR
  FPR was 0.14 vs legacy 0.05 on the same slice, so it is not the default.
- **JPEG70 LSB** on the locked slice: TPR@5%FPR 0.10 (spatial LSB dies).
  Frequency DCT TPR@5%FPR 0.86 vs identity 1.00. Patchwork identity
  TPR@5%FPR 0.64 so the YAML weight stays.

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

v0.3.1 recovers **keyless plaintext** and **localizes** LSB patches:

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
