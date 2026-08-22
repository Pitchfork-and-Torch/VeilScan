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
- **Decision threshold 0.48** is still the default until
  `operating_point.status=locked` (n>=50). A provisional n=20 generator-photo
  identity+jpeg_70 slice suggested ~0.67 to hold FPR near 0.05 (cover mean
  ~0.65). Do not quote that as camera FPR. Fusion mode `specialist_or` is
  implemented but not the default until that lock.
- **JPEG70 LSB** on that slice: TPR@5%FPR ~0.05 (spatial LSB dies). Frequency
  DCT TPR@5%FPR ~0.70 vs identity 1.00 (retention bar). Patchwork identity
  TPR@5%FPR ~0.70 so the YAML weight stays for now.

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
