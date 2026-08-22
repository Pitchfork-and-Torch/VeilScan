# Next upgrade: v1.1 native JPEG path

**Status:** SHIPPED in v1.1.0. Next leftover: JPEG50 frequency TPR (DCT 0.18).
v0.4 operating-point work shipped (historical
notes stay in `docs/UPGRADE_PLAN.md` and older RESULTS). v1.0.0 shipped
JPEG-hardened FSNet + pixel `jpeg_like` blend.

**Identity:** presence ensemble plus keyless plaintext decode. No remover.
No SynthID verifier. No UniFreq in git. Generator `present` stays locked.

Lamp: https://veilscan.jonbailey.xyz/

---

## 1. Why this upgrade

v1.0 blends toward FSNet when the **decoded array** looks blocky
(`jpeg_blockiness >= 1.10`). Camera stills are already JPEG files. Resize
and mild Q-tables often sit under that cut, so identity camera plates miss
the blend they were supposed to get.

JPEG70 DWT TPR@5%FPR on BSDS is still 0.30. That is a cook/attack problem.
The product hole in front of it is simpler: **know that the file is JPEG**
without waiting for 8x8 ringing.

**North star for v1.1:** `veilscan scan photo.jpg` sets `jpeg_container`
from the bytes, estimates Q-table quality, names chroma subsampling, and
turns `jpeg_like` on so the v1.0 blend actually fires on camera JPEGs.
JPEG50 is **measured** on the same camera pack. Default `present` does
not flip.

## 2. Do not break

- Generator lock `op-v0.4.0-locked-n50` ~0.67. Never overwrite from `--covers`.
- FSNet cook: no `lsb`. No frozen BSDS test pack (`data/covers/camera`).
- ResidualCNN unchanged.
- Patchwork YAML weight 0.
- Fusion `legacy` unless nested OR FPR also holds (v1.0 nested OR FPR 0.20
  vs legacy 0.12: do not flip).
- No remover. No Gradio from a Grok Build command.
- Schema 3 fields stay (`jpeg_like`, `jpeg_blockiness`). Schema 4 **adds**
  fields; it does not rename.

## 3. Feature (this cut)

| Piece | Truth |
|-------|--------|
| JPEG SOI / SOF / DQT parse | Local, no `jpeglib` required |
| `jpeg_container` | True when file bytes start `FF D8` |
| `jpeg_quality_est` | IJG-style estimate from luma DQT, or null |
| `jpeg_subsampling` | `4:4:4` / `4:2:2` / `4:2:0` / null |
| `jpeg_like` | `jpeg_container` **or** blockiness >= 1.10 |
| PNG / BMP / TIFF | container false; blend only if the array is blocky |
| JPEG50 | extra attack on the camera bench; **not** in the OP lock slice |
| Camera sidecar | re-lock n=50 after container blend. New id `op-v1.1.0-camera-locked-n50` |
| DIV2K | optional confirmation (`--corpus-id camera-div2k`). Those files are PNG. |

Lock slice stays `camera/identity` + `camera/jpeg_70` so the cut stays
comparable to v1.0. JPEG50 lives in `cells`, not in `operating_slice`.

## 4. Acceptance

- [x] `analyze_path` on a JPEG sets `jpeg_container` true and `schema_version` 4
- [x] sine PNG cover is not `jpeg_like`; JPEG70 roundtrip is
- [x] Q50 estimate is lower than Q90 on the same plate
- [x] Camera n=50 sidecar rewritten; generator lock file untouched
- [x] Nested OR still not default
- [x] Pytest green
- [x] Lamp mast + JSON-LD softwareVersion 1.1.0. Site cache-bust **1.4.0**
      (do not reuse `?v=1.1.0`; that URL already meant an older OG card)
- [x] Latest-only private GitHub release `v1.1.0`

## 5. Not this cut

- Recook FSNet for DWT-JPEG (next science if JPEG50/DWT stays ~0.30 after
  the blend actually fires)
- Flip default `present` to the camera sidecar
- UniFreq-100K
- In-tab presence scores on the lamp
- Patchwork keyed specialist
