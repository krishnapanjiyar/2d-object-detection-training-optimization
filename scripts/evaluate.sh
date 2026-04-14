#!/bin/bash
# =============================================================
# Evaluate both baseline and optimized models on COCO val2017
# =============================================================

set -e

echo "============================================"
echo "  Evaluating Baseline Model"
echo "============================================"

BASELINE_CONFIG="configs/baseline_faster_rcnn_r50_fpn.py"
BASELINE_CKPT="work_dirs/baseline_r50_fpn_1x/best_coco_bbox_mAP_epoch_12.pth"

python tools/test.py \
    "$BASELINE_CONFIG" \
    "$BASELINE_CKPT" \
    --out "results/baseline_results.pkl" \
    2>&1 | tee results/baseline_eval.log

echo ""
echo "============================================"
echo "  Evaluating Optimized Model"
echo "============================================"

OPTIMIZED_CONFIG="configs/optimized_faster_rcnn_r101_pafpn.py"
OPTIMIZED_CKPT="work_dirs/optimized_r101_pafpn_2x/best_coco_bbox_mAP_epoch_24.pth"

python tools/test.py \
    "$OPTIMIZED_CONFIG" \
    "$OPTIMIZED_CKPT" \
    --out "results/optimized_results.pkl" \
    2>&1 | tee results/optimized_eval.log

echo ""
echo "============================================"
echo "  Generating Comparison"
echo "============================================"

python scripts/compare_results.py

echo "Done! See results/comparison_table.md"
