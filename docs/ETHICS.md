# Ethics and responsible use

VeilScan is a **local detector** for invisible image watermarks. Intended uses:

- copyright / provenance screening
- forensic triage ("is there a mark?" before calling a specific decoder)
- dataset hygiene (filter watermarked training data)
- research on agnostic presence detection (AWPD)

## Dual use

The same signals that detect a mark can, in other hands, help *strip* one.
This repository:

- does **not** ship a removal / purification attack
- ships embedders only as evaluation fixtures for the detector
- will not add a "clean this image" mode

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
