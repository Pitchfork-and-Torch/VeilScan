# VeilScan research survey (Phase 0)

Living notes for **agnostic invisible-watermark presence detection** (AWPD):
does this image carry an invisible watermark, without a matching decoder or key.

Product name: **VeilScan**. Question it answers: "Does this image contain an invisible watermark?"

Last updated: 2026-08-20.

## 1. Problem

Invisible watermarks hide a mark under a perceptual budget `D(x_w, x_c) <= eps`.
Families do not share a decoder:

| Family | Where the mark lives | Typical decoder |
|--------|----------------------|-----------------|
| LSB / Patchwork | bit planes, pair statistics | keyless stats or key |
| DCT / DWT / SVD / hybrid | transform coefficients | matching inverse + key |
| HiDDeN / StegaStamp / RivaGAN | learned spatial residual | trained decoder |
| Stable Signature | fine-tuned VAE decoder | secret decoder |
| Tree-Ring | diffusion latent FFT rings | DDIM inversion + FFT |
| Gaussian Shading | latent random seed / bits | recovery of seed |
| SynthID-style | proprietary multi-scale | closed verifier |

Decoding an unknown scheme is ill-posed. **Presence detection** is the degenerate
task: binary `y in {0,1}` from traces that survive the HVS masking constraint.

Two operating modes:

1. **Single-image, zero-shot** (VeilScan default). No clean twin, no algorithm ID.
2. **Dataset vs clean reference** (WMD). Annotation-free, still algorithm-agnostic.

Leave-one-algorithm-out is the honest protocol (UniFreq-100K / AWPD). Random mixed
splits overstate generalization.

## 2. What actually works (and what does not)

### 2.1 Classical spatial / statistical

- **Chi-square (Westfeld-Pfitzmann, 1999).** Sequential LSB equalizes adjacent
  histogram bins `(2k, 2k+1)`. High p-value => LSB randomization. Weak on
  scattered / low-rate embedding.
- **RS analysis (Fridrich, Goljan, Du 2001).** Regular vs singular groups under
  LSB flip masks. Best cheap detector for random LSB.
- **Sample pair analysis (Dumitrescu et al.).** Adjacent-pair transition
  asymmetries. Complements RS.
- **Bit-plane entropy / correlation.** LSB plane of a natural photo is structured;
  a payload flattens it.
- **Patchwork (Bender et al. 1996).** Pair-brightness tests. AWPD paper: models
  fail here and on LSB (accuracy < 0.60 leave-one-out) because the cue is
  combinatorial, not a high-frequency texture.
- **SRM (Fridrich and Kodovsky, 2012).** High-pass residuals + co-occurrence.
  Designed for low-payload steganography. Still a strong *feature* for spatial marks.
  Full 34k-dim SRM is heavy; VeilScan ships a compact residual + co-occurrence core.

Implication: keep **bit-plane / RS / chi-square / SPA** in the ensemble. Deep
frequency nets do not replace them.

### 2.2 Frequency and transform domain

HVS masking pushes many *robust* marks into mid/high frequency, but not all marks.

- **Block DCT (8x8).** Cox spread-spectrum, JPEG-domain marks, many hybrids.
  Natural AC coefficients are approximately Laplacian. Extra mid-band energy,
  histogram peakiness, and block-wise energy maps are useful.
- **DWT / Haar packets.** LH/HL/HH energy ratios and kurtosis. Hybrid DWT-DCT-SVD
  (Navas et al. 2008 / "DDS") is a common robust recipe.
- **DFT magnitude and phase.** Radial 1/f spectrum of photos. Excess high-frequency
  energy, phase structure, and **circular rings** (Tree-Ring style) show up here.
- **Fourier-Mellin (log-polar FFT).** Rotation/scale-tolerant spectral signatures.
- **DCTR / PHARM / GFR-style histograms.** Compact frequency residual features
  from the steganalysis literature.

FSNet/AWPD (Ao et al., arXiv:2603.06723, CVPRW 2026 SAFE):

- Task: AWPD. Dataset: UniFreq-100K (LSB, Patchwork, DCT, DWT, HiDDeN, StegaStamp,
  Stable Signature, Tree-Ring, SynthID).
- **ASPM:** learnable 2D-DCT gate in the stem. Amplify watermark bands, suppress
  low-frequency semantics *before* downsampling throws them away.
- **DMSA:** multi-basis DCT channel attention + tri-stream (avg/max/min) pooling
  so both energy peaks and valleys count.
- Leave-one-algorithm-out: strong on DCT/DWT/HiDDeN/StegaStamp/SSig/Tree-Ring/SynthID;
  **fails on LSB and Patchwork** (same as ResNet/ViT/DINOv2).
- Official weights were not public at authoring. VeilScan reimplements the *ideas*
  (ASPM + DMSA + residual CNN), not a private checkpoint.

### 2.3 Deep steganalysis

- **Xu-Net, YeNet, SRNet (Boroumand, Chen, Fridrich 2018), CovNet.** Residual
  CNNs that keep noise. SRNet is the usual strong spatial-domain baseline.
- These expect *stego vs cover* of a known embedding family. Zero-shot to
  Tree-Ring / SynthID is not free. Train on a diverse synthetic mix, then
  leave-one-family-out.

VeilScan: compact residual CNN with optional fixed high-pass first layer.
Untrained weights are **excluded from the ensemble** (weight 0).

### 2.4 Black-box reference (WMD)

Pan et al., ECCV 2024, arXiv:2403.15955. **WaterMark Detector (WMD)**:

- Needs a *clean reference set* of similar visual distribution, plus a suspect set.
- Offset learning: cancel shared clean gradients; leftover gradient is the mark.
- Asymmetric loss: softmax/temperature on clean (strict min) + linear on suspects
  (do not let abundant clean suspects dominate).
- Iterative pruning: drop lowest-scoring suspects, re-init, repeat until ~5% remain.
- Reported AUC > 0.9 on many single-watermark 5%-prevalence sets; > 0.7 multi-mark.
- Weakest on LSB and Tree-Ring among their six methods.
- **Not a single-image detector.** VeilScan exposes it only with `--reference-dir`.

We do not implement the paper's "watermark removal" appendix. Detection only.

### 2.5 Foundation models and anomaly

VFMs (ResNet, ViT, CLIP, DINOv2, ConvNeXt) have **low-frequency bias** (Yin et al.,
Fourier perspective on robustness). AWPD table: they work on some frequency marks
and collapse on LSB/Patchwork. Useful as an optional anomaly head, not as the
primary detector.

Cheap local stand-ins (no mandatory download):

- Patch-PCA / autoencoder reconstruction residual
- Denoising residual (Wiener / Gaussian / bilateral) + higher-order stats
- DIRE-like idea for suspected diffusion images only if an inversion backend exists
  (not wired by default)

### 2.6 Generative-AI watermarks (detection cues, not decoders)

- **Tree-Ring (Wen et al. 2023).** Rings in inverted latent FFT. Pixel-space cue:
  weak circular energy in the image FFT after 1/f removal. Inversion would be
  stronger and is optional later.
- **Stable Signature (Fernandez et al. 2023).** Decoder-side; pixel residual is a
  learned high-frequency pattern. Frequency + residual CNNs transfer better than LSB tests.
- **Gaussian Shading.** Mostly latent; pixel-only detection is weak. Be honest.
- **SynthID-image (Gowal et al. 2025, arXiv:2510.09263).** Closed. Treat as
  unknown high-frequency / multi-scale residual. No claim of a SynthID verifier.

### 2.7 Attacks the evaluator must run

JPEG QF sweep, resize, crop, additive noise, blur, color jitter, and (when a
diffusion backend exists) regeneration / purification. Deep marks are trained
against some of these; LSB is not.

## 3. Ensemble policy

Complementary failure modes:

| Cue | Strong on | Weak on |
|-----|-----------|---------|
| RS / chi-square / SPA / bit-plane | LSB | Tree-Ring, SSig |
| DCT / DWT / hybrid | classical frequency, DDS | pure bit flips |
| Circular FFT | Tree-Ring-like | LSB |
| SRM residual | spatial / learned residual | very low amplitude latent |
| FSNet-like / residual CNN | modern high-freq marks (if trained) | LSB, Patchwork, untrained |
| WMD | dataset-level unknown marks | no reference set, single image |

Fusion: reliability-weighted mean of *active* detectors, disagreement as
uncertainty. Untrained or inapplicable modules contribute 0 weight, not 0.5 noise.

Progressive tiers: `fast` (spatial + cheap FFT/DCT) -> `frequency` -> `deep`
(if checkpoints exist) -> `blackbox` (if reference given).

## 4. Open source vs first principles

| Component | Decision |
|-----------|----------|
| Chi-square, RS, SPA, Patchwork, DCT/DWT/DFT, Haar | first principles |
| Compact SRM residual + co-occurrence | first principles (not full SRM) |
| ASPM / DMSA / FSNet | reimplement paper ideas; no private weights |
| WMD offset + prune | reimplement paper algorithm, small CNN |
| HiDDeN / StegaStamp / RivaGAN official embedders | optional later; synthetic approximations for tests |
| SynthID / closed APIs | never reverse a private verifier; treat as unknown |

## 5. Computational honesty (this machine)

- GPU: NVIDIA RTX 4080 present.
- Default PyTorch at authoring: **CPU wheel**. Deep training is optional and slow
  until a CUDA build is installed. Classical + frequency ensemble is the
  production default and is real-time on CPU for megapixel images with tiling.

## 6. Citations (identifiers)

- Westfeld and Pfitzmann, IH 1999 (chi-square LSB).
- Fridrich, Goljan, Du, ACM Multimedia 2001 (RS).
- Dumitrescu, Wu, Memon (sample pair analysis).
- Bender, Gruhl, Morimoto, Lu, IBM SJ 1996 (Patchwork).
- Cox, Kilian, Leighton, Shamoon, IEEE TIP 1997 (spread spectrum).
- Fridrich and Kodovsky, IEEE TIFS 2012 (SRM).
- Boroumand, Chen, Fridrich, IEEE TIFS 2018 (SRNet).
- Zhu et al., HiDDeN (2018).
- Tancik, Mildenhall, Ng, CVPR 2020 (StegaStamp).
- Navas et al., COMSWARE 2008 (DWT-DCT-SVD).
- Fernandez et al., ICCV 2023 (Stable Signature).
- Wen et al., 2023 (Tree-Ring).
- Pan et al., ECCV 2024 / arXiv:2403.15955 (WMD).
- Ao et al., arXiv:2603.06723 (AWPD / FSNet / UniFreq-100K).
- Gowal et al., arXiv:2510.09263 (SynthID-image).
- Yin et al., NeurIPS 2019 (Fourier perspective / low-frequency bias).

## 7. Phase 0 decisions locked

1. Name: VeilScan. Package: `veilscan`.
2. Local-first. No mandatory cloud, no mandatory VFM download.
3. Plugin detectors, common `DetectionResult`.
4. Generators exist **only** to train/evaluate detectors, not as a product feature
   for hiding data.
5. No watermark *removal* pipeline.
6. Deep modules stay silent in the ensemble until a checkpoint is present.
7. Success bar for v0: classical+frequency ensemble demonstrably separates
   synthetic LSB / DCT / DWT / tree-ring-approx / patchwork from matched covers,
   with tests and a CLI.
