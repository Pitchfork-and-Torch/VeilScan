# Ethics and responsible use

VeilScan is a **local detector** for invisible image watermarks and a
**forensic hunt extractor** for authorized files. Intended uses:

- copyright / provenance screening
- forensic triage ("is there a mark?")
- keyless plaintext recovery (sequential LSB, PNG text, JPEG comments) on files you are authorized to inspect
- CTF / DFIR hunt extract (`veilscan hunt`) on files you are authorized to inspect
- dataset hygiene (filter watermarked training data)
- research on agnostic presence detection (AWPD)

## Dual use

The same signals that detect a mark can, in other hands, help *strip* one.
This repository:

- does **not** ship a removal / purification attack
- ships embedders and `embed-text` only as evaluation fixtures
- will not add a "clean this image" mode
- decode prints keyless plaintext only; it is not a universal watermark decoder
- hunt extracts container text, trailing/embedded files, and FLAG{} on authorized inputs
- hunt may later brute known-tool passphrases against a wordlist the operator supplies; do not vendor rockyou

Do not use VeilScan to bypass copyright, provenance, or C2PA-style
authenticity systems. Do not use the generators to hide unauthorized data
in other people's images.

## Limits you must state in any report

- Absence of a detection is **not** proof the image is unmarked (SynthID,
  Gaussian Shading, low-payload LSB, and latent-only marks are easy to miss).
- Presence is a **score**, not a legal identification of an owner or model.
- Scores are sensitive to distribution shift (screenshots, heavy JPEG, memes).
- Deep modules without a trained checkpoint do not contribute.

## Privacy

Default path is local. No pixels are uploaded. Optional foundation-model
weights, if the operator later enables them, stay on disk.

## Attribution

When adapting published methods (RS, SRM, FSNet ideas, WMD offset learning),
cite the papers listed in `docs/RESEARCH.md`. This is an independent
reimplementation of the ideas, not the authors' official code.
