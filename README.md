## Best Model Performance

- **Loss Function:** BCE-Lovász
- **Monitor Metric:** F1
- **Weight Decay:** 1e-2

### Siamese U-Net with SMP Backbone and Multi-Scale CBAM Fusion (`smp_siamese`)

| AUC-ROC | F1     | IoU    | Val. Loss | Train Loss |
| ------- | ------ | ------ | --------- | ---------- |
| 0.9994  | 0.8777 | 0.7821 | 0.3390    | 0.3502     |

### Custom Siamese U-Net with Bottleneck Attention (`custom_unet`)

| AUC-ROC | F1     | IoU    | Val. Loss | Train Loss |
| ------- | ------ | ------ | --------- | ---------- |
| 0.9994  | 0.8936 | 0.8076 | 0.3155    | 0.3484     |
