# VeilScan v2 hunt contract

**Identity:** `veilscan hunt` extracts hidden payloads from authorized
image/container files. `veilscan scan` stays the AWPD presence module.
No watermark remover.

Package: `2.1.0`. Private GitHub. Proprietary license. Public lamp stays
generic marketing (`veilscan.jonbailey.xyz`) and does not document extractors.

## Parked leftover (do not recook)

JPEG50 DCT ensemble TPR at 128px is 0.46. Eval amp 14 vs Q50 luma steps
~51/56. v1.8 inspect prints those steps. Do not recook FSNet. Do not flip
default `present` (~0.67). Do not replace the 128px BSDS camera lock
(~0.756). No from-scratch Q50. No `specialist_or` fusion. No LSB on FSNet.

That leftover is not the income path.

## v2.0.0 shipped (Phase 0 + Phase 1)

- `veilscan hunt FILE [--json] [--out DIR] [--flag-re] [--stop-on-flag]`
- `veilscan gym` plants four eval fixtures and scores extract TPR
- Native carve: trailing after PNG IEND / JPEG EOI, ZIP/PDF/RAR/7z/gzip magics
- All PNG text and unknown ancillary chunks (no PREFERRED_KEYS drop on hunt)
- JPEG COM + APPn + EXIF; GIF comments; WEBP chunks
- strings + FLAG{} harvest
- Decode token rail: short photo LSB `TAG:VAL` (PTQ:B7C) no longer `found`

Gym bar for this slice: 4/4 unaided, no Linux binaries.

## v2.1.0 shipped (Phase 2)

- Native zsteg-class walks: r/g/b/a/rgb/bgr/rgba x bits 0-3 (0-7 with `--deep`) x msb/lsb x row/col/snake
- zlib / file-magic / OPENSTEGO / CAMOUFLAGE harvest on packed bitstreams
- Palette-index LSB before PIL expand
- Bitplane sheet (`bitplanes.png`) + OpenCV QR on bit planes
- Gym +4: `zsteg-rgb-bit0`, `alpha-lsb`, `palette-lsb`, `bitplane-qr` (8/8)

Ruby zsteg is not imported. No Java Stegsolve.

## Still later (do not pretend they shipped)

3. JSteg gym path; optional stegseek adapter; `--wordlist` (no rockyou in git)
4. Hunt HUD PNG + agent JSON polish
5. WAV LSB + spectrogram stills
6. n>=20 gym TPR >= 0.90, then one live unseen CTF image as external proof

## Rails

- Authorized files only. Extract is not strip.
- Generators / gym plants are eval-only.
- Optional PATH adapters never required for core gym.
- No Gradio/uvicorn from Grok Build.
- No Volatility, PCAP, or disk images in v2.
