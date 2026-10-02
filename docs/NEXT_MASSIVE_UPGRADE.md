# VeilScan v2 hunt contract

**Identity:** `veilscan hunt` extracts hidden payloads from authorized
image, WAV, MP3, FLAC, PDF, and container files. `veilscan scan` stays the AWPD presence module.
No watermark remover.

Package: `2.6.0`. Public, source-available repo (see `LICENSE`).

## Parked leftover (do not recook)

JPEG50 DCT ensemble TPR at 128px is 0.46. Eval amp 14 vs Q50 luma steps
~51/56. v1.8 inspect prints those steps. Do not recook FSNet. Do not flip
default `present` (~0.67). Do not replace the 128px BSDS camera lock
(~0.756). No from-scratch Q50. No `specialist_or` fusion. No LSB on FSNet.

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

## v2.2.0 shipped (Phase 3)

- Native JSteg DCT AC LSB plant+extract via `jpeglib` (pip extra, not a PATH binary)
- `--wordlist` for PATH adapters: stegseek, steghide, outguess, jpseek
- Missing adapter = skip note; gym required TPR still 1.0
- Gym `jsteg-flag.jpg` when jpeglib present; `steghide-password.jpg` when steghide plants
- F5/nsF5 extract is a note, not a claim
- No rockyou in git. No 2^32 seed brute in Python (stegseek --seed if on PATH)

## v2.3.0 shipped (Phase 4)

- `hunt --out DIR` writes `report.png` (HUD) + `findings.json` (schema 2) + carved blobs
- JSON: `status`, `summary`, `flag_count`, `finding_count` so an agent can parse flags without the HUD
- CLI default out dir `{stem}-veilscan-hunt` next to the file; `--no-report` skips it
- `--stop-on-flag` already shipped in v2.0

## v2.3.1 shipped (inspect stamp + PNG compressed text)

- inspect JSON `jpeg.dct_pair_34_43` next to luma_q(3,4)/(4,3); CLI prints the pair
- Keyless decode plants/reads PNG zTXt and iTXt. Gym `png-zTXt-flag` + `png-iTXt-flag`
- Score mix and present cut unchanged. No stripper.

## v2.4.0 shipped (signal case)

Phase 5, plus two hunt surfaces the single-file image loop did not have.

- `veilscan hunt FILE.wav` reads PCM 8/16 mono and stereo sample LSB (per channel, interleaved, forward and reversed, bit 0; `--deep` adds bits 1-2)
- Spectrogram still: `spectrogram.png` in `--out`. Magnitude only. No OCR
- Automated frequency hits: `VSPEC1` exact-bin bytes, and a QR drawn as exact-bin tones
- One-level nested hunt: a zip member or trailing blob with no plaintext FLAG is hunted again when it is an image or WAV (compressed PNG zTXt inside a zip)
- `veilscan case DIRECTORY` writes `case.json` (schema 1) with one flag list across images and WAVs
- Gym required count is at least 20 and required TPR stays 1.0. New required rows: gif comment, webp XMP, BMP LSB, column LSB, four WAV LSB layouts, tone bytes, zip-wrapped zTXt. Spectrogram QR is required when OpenCV can encode it
- Flag harvest keeps printable ASCII matches only, so a noisy bit plane cannot mint a second binary FLAG{}
- Presence OP unchanged. No stripper. No FSNet recook

## v2.5.0 shipped (PDF hunt)

The v2.4.0 loop sniffed a PDF as `unknown`, `scan` threw a PIL error, and `hunt` reported the file's own `%PDF` header as a carved PDF.

- `sniff_kind` returns `pdf`. `scan` refuses a PDF and names `veilscan hunt`. Keyless image `decode` does not pretend to read one
- `veilscan hunt FILE.pdf` reads Info, JavaScript, URI and Launch actions, embedded files, rendering-mode-3 text, comments, bytes after the last `%%EOF`, and objects replaced by a later revision
- Embedded images and files are handed to the existing one-level image/WAV/PDF hunt. Flate image samples are rebuilt when they are 8-bit gray or RGB
- Visible page text is not a finding. Carve skips the file's own PDF header and magics that sit inside PDF streams
- `veilscan case` includes `.pdf`
- Gym required rows added: info, flate JavaScript, flate embedded file, trailing bytes, comment, flate invisible text, incremental revision, nested PNG zTXt. Required TPR stays 1.0
- No stripper. No object-stream or xref-stream expander. No white-text claim. Presence OP unchanged

## v2.6.0 shipped (compressed audio)

- `sniff_kind` returns `mp3` and `flac`. `scan` refuses both and names `veilscan hunt`
- ID3 text and a FLAC Vorbis comment are read with the stdlib. UTF-16 ID3 is a real hit, not an ASCII strings hit
- Optional extra `audio` (`miniaudio`) decodes PCM into the v2.4.0 LSB, tone, and QR walks. Core gym TPR does not depend on it
- Verbatim FLAC keeps planted 16-bit sample LSB. MP3 sample LSB is not a gym row
- Letters painted into the spectrogram are read with `tesseract` when it is on PATH. The magnitude still is still written
- `veilscan case` includes `.mp3` and `.flac`
- Gym rows: `mp3-id3`, `flac-comment`, `flac-lsb` (when miniaudio imports), `wav-spectrogram-ocr` (when tesseract is on PATH). Required TPR stays 1.0
- No stripper. No F5 extract. No object-stream expander. No unseen CTF image. Presence OP unchanged

## Still later (do not pretend they shipped)

6. One live unseen CTF image as external proof (gym n>=20 and TPR 1.0 already hold)
8. F5/nsF5 extract (still a note)

## Rails

- Authorized files only. Extract is not strip.
- Generators / gym plants are eval-only.
- Optional PATH adapters never required for core gym.
- No unattended Gradio or uvicorn server.
- No Volatility, PCAP, or disk images in v2.
