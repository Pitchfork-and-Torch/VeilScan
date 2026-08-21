# Limitations (honest)

## Detection

- **LSB and Patchwork** vs modern nets: AWPD/FSNet leave-one-out accuracy was
  below 0.60. VeilScan therefore keeps classical bit-plane tests. Those tests
  themselves fail on latent-only generative marks.
- **Tree-Ring / Gaussian Shading / SynthID** in pixel space without inversion
  or a vendor verifier is weak. The circular-FFT detector is an approximation
  of Tree-Ring, not a replacement for DDIM inversion.
- **Single-image WMD** is undefined. WMD needs a clean reference set.
- **Untrained CNNs** would add noise. They are gated off until
  `checkpoints/*.pt` exist.
- **Calibration** of scores to real-world false-positive rates is not done
  against UniFreq-100K (dataset not bundled). v0 calibration is synthetic.

## Robustness

JPEG, downscale, and diffusion regeneration destroy or hide many spatial marks.
A high score on a pristine PNG can collapse after social-media re-encode.

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

A small LSB rectangle is sequential only inside that box. The decoder hunts
low bit-0 autocorrelation tiles, then tries those windows. It does not need
the coordinates in advance.

It will not print encrypted stego (Steghide / OpenStego with a password),
SynthID, Digimarc, Tree-Ring payloads, or HiDDeN-class neural marks.
Spatial LSB usually dies after JPEG / Telegram recompress. Hotspots can
still mark the patched region. A recovered string is not proof of authorship.

## Legal

A VeilScan score is not a copyright determination and not evidence of a
specific generator.
