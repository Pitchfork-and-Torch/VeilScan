# VeilScan bench

protocol `veilscan-bench-v1` n=50 size=128 seed=20260822

generator-photo / generator-sine. Do not cite as camera-photo FPR.

| slice | family | AUC | TPR@5%FPR | mean cover | mean marked |
|-------|--------|-----|-----------|------------|-------------|
| photo/identity | lsb | 1.000 | 1.000 | 0.644 | 0.761 |
| photo/identity | dct | 1.000 | 1.000 | 0.644 | 0.769 |
| photo/identity | dwt | 1.000 | 1.000 | 0.644 | 0.784 |
| photo/identity | spread | 1.000 | 1.000 | 0.644 | 0.791 |
| photo/identity | tree_ring | 0.985 | 0.980 | 0.644 | 0.754 |
| photo/identity | patchwork | 0.964 | 0.640 | 0.644 | 0.689 |
| photo/jpeg_70 | lsb | 0.600 | 0.100 | 0.509 | 0.517 |
| photo/jpeg_70 | dct | 0.965 | 0.940 | 0.509 | 0.660 |
| photo/jpeg_70 | dwt | 0.894 | 0.700 | 0.509 | 0.587 |
| photo/jpeg_70 | spread | 1.000 | 1.000 | 0.509 | 0.731 |
| photo/jpeg_70 | tree_ring | 1.000 | 1.000 | 0.509 | 0.731 |
| photo/jpeg_70 | patchwork | 0.922 | 0.600 | 0.509 | 0.560 |

## Operating point (derived)

- id: `op-v0.4.0-locked-n50` status `locked`
- threshold=0.669729 t_lsb=0.229836 t_freq=0.179105 t_class=0.908002
- fpr_est=0.05 n=50
- fusion_mode=legacy

## A/B legacy vs specialist-OR

- legacy_fpr=0.05 or_fpr=0.14
- LSB TPR legacy=0.5 or=0.61
- DCT TPR legacy=0.79 or=1.0
- flip_default=False

- nested n_fit=50 n_eval=50 legacy_fpr=0.04 or_fpr=0.22 flip=False

