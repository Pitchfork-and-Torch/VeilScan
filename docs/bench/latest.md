# VeilScan bench

protocol `veilscan-bench-v1` n=20 size=128 seed=20260822

generator-photo / generator-sine. Do not cite as camera-photo FPR.

| slice | family | AUC | TPR@5%FPR | mean cover | mean marked |
|-------|--------|-----|-----------|------------|-------------|
| photo/identity | lsb | 1.000 | 1.000 | 0.645 | 0.763 |
| photo/identity | dct | 1.000 | 1.000 | 0.645 | 0.769 |
| photo/identity | dwt | 1.000 | 1.000 | 0.645 | 0.780 |
| photo/identity | spread | 1.000 | 1.000 | 0.645 | 0.791 |
| photo/identity | tree_ring | 0.965 | 0.950 | 0.645 | 0.745 |
| photo/identity | patchwork | 0.960 | 0.700 | 0.645 | 0.692 |
| photo/jpeg_70 | lsb | 0.655 | 0.050 | 0.502 | 0.512 |
| photo/jpeg_70 | dct | 0.927 | 0.700 | 0.502 | 0.630 |
| photo/jpeg_70 | dwt | 0.892 | 0.550 | 0.502 | 0.583 |
| photo/jpeg_70 | spread | 1.000 | 1.000 | 0.502 | 0.721 |
| photo/jpeg_70 | tree_ring | 1.000 | 1.000 | 0.502 | 0.724 |
| photo/jpeg_70 | patchwork | 0.922 | 0.650 | 0.502 | 0.559 |

## Operating point (derived)

- id: `op-v0.4.0-provisional-n20` status `provisional`
- threshold=0.674216 t_lsb=0.224773 t_freq=0.135945 t_class=0.907289
- fpr_est=0.05 n=20

