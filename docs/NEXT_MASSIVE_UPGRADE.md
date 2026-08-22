# Next leftover after v1.3.0

**Status:** v1.3.0 SHIPPED. JPEG50 DCT ensemble TPR is 0.46 (was 0.18 at
v1.1, 0.32 at v1.2). Still not a solved family.

Lamp: https://veilscan.jonbailey.xyz/ now lists inspect, JPEG container
fields, ResidualCNN vs FSNet, doctor, and honest Q50 DCT limits.

## Do not break

- Generator lock `op-v0.4.0-locked-n50` ~0.67. Never overwrite from `--covers`.
- FSNet cook: no `lsb`. No frozen BSDS test. No from-scratch Q50.
- ResidualCNN unchanged unless a measured LSB hole appears.
- Patchwork weight 0. Fusion `legacy` unless nested OR FPR also holds.
- No remover. No Gradio from a Grok Build command.

## Optional next

Another DCT-Q50 fine-tune, or stop and leave DCT at Q50 labeled weak.
Do not flip default `present`. DIV2K sidecar is still the v1.0 PNG pack.
