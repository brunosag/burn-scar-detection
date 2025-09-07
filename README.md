## Best Model Performance

### Siamese U-Net with SMP Backbone and Multi-Scale CBAM Fusion (`smp_siamese`)

| AUC-ROC | F1     | IoU    | Val. Loss | Train Loss |
| ------- | ------ | ------ | --------- | ---------- |
| 0.9993  | 0.9004 | 0.8188 | 0.3225    | 0.3225     |

- **Loss Function:** BCE-Lovász (single-stage)
- **Monitor Metric:** F1
- **Weight Decay:** 1e-2
- **Learning Rate:** 1e-2

### Custom Siamese U-Net with Bottleneck Attention (`custom_unet`)

| AUC-ROC | F1     | IoU    | Val. Loss | Train Loss |
| ------- | ------ | ------ | --------- | ---------- |
| 0.9994  | 0.8889 | 0.8000 | 0.3278    | 0.3581     |

- **Loss Function:** BCE-Lovász (single-stage)
- **Monitor Metric:** F1
- **Weight Decay:** 1e-2
- **Learning Rate:** 3e-2
