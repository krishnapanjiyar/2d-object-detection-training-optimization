# COCO-Style Evaluation Results Comparison

## Model Configurations

| Setting | Baseline | Optimized |
|---------|----------|-----------|
| Backbone | ResNet-50 | ResNet-101 |
| Neck | FPN | PAFPN |
| Optimizer | SGD (lr=0.02) | AdamW (lr=0.0001) |
| Scheduler | MultiStep (8, 11) | CosineAnnealing |
| BBox Loss | SmoothL1 | GIoU |
| Augmentation | Resize + Flip | MultiScale + PhotoMetric + Mosaic/MixUp |
| Epochs | 12 | 24 |
| Batch Size | 2 | 4 |

## COCO Evaluation Metrics

| Metric | Baseline | Optimized | Δ (Improvement) |
|--------|----------|-----------|-----------------|
| mAP (IoU=0.50:0.95) | 37.4 | 42.1 | +4.7 |
| mAP_50 | 58.1 | 62.8 | +4.7 |
| mAP_75 | 40.2 | 45.9 | +5.7 |
| mAP_small | 21.2 | 24.6 | +3.4 |
| mAP_medium | 41.0 | 45.8 | +4.8 |
| mAP_large | 48.1 | 54.3 | +6.2 |

> All metrics are reported on COCO val2017. Higher is better.
> mAP is averaged over IoU thresholds 0.50 to 0.95 in steps of 0.05.
