# VeilScan v0.2.3

FSNet-lite now concatenates RGB + LSB planes (same lesson as ResidualCNN) and uses mean+std pooling. Micro overfit on 6 images: LSB and DCT both reach acc 1.0.

## Live n=6 mix pair means

| detector | family | cover | marked | delta |
|----------|--------|-------|--------|-------|
| residual_cnn | lsb | 0.079 | 0.989 | +0.91 |
| residual_cnn | dct | 0.079 | 0.393 | +0.31 |
| residual_cnn | spread | 0.079 | 0.282 | +0.20 |
| residual_cnn | tree_ring | 0.079 | 0.356 | +0.28 |
| residual_cnn | patchwork | 0.079 | 0.267 | +0.19 |
| residual_cnn | dwt | 0.079 | 0.127 | +0.05 |
| fsnet_lite | lsb | 0.399 | 0.970 | +0.57 |
| fsnet_lite | dct/spread/dwt | 0.399 | ~0.399 | ~0 |

FSNet val AUC 0.585: it is an LSB specialist, not a general frequency net. Still added to `peak_ok` because cover stays ~0.40 (under 0.48) and LSB is the family frequency nets historically miss.

ResidualCNN continued 200 steps on more families; val AUC 0.80. Cover score dropped to 0.079.

## Train recipe

```powershell
py -3 scripts\train_lite.py --fresh --only fsnet_lite --steps 400 --families lsb,dct,spread,dwt --jpeg-prob 0.2
py -3 scripts\train_lite.py --only residual_cnn --steps 200 --families lsb,dct,spread,dwt,tree_ring,patchwork --jpeg-prob 0.2 --lr 0.0005
```

Needs `torch` CUDA (`2.13.0+cu130` on this machine).
