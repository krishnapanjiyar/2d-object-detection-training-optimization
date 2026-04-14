#!/bin/bash
# =============================================================
# Train Optimized: Faster R-CNN with ResNet-101-PAFPN + improvements
# =============================================================

set -e

CONFIG="configs/optimized_faster_rcnn_r101_pafpn.py"
WORK_DIR="work_dirs/optimized_r101_pafpn_2x"
GPUS=${GPUS:-1}

echo "============================================"
echo "  Training Optimized Model"
echo "  Config:   $CONFIG"
echo "  Work Dir: $WORK_DIR"
echo "  GPUs:     $GPUS"
echo "============================================"

# Single GPU training
if [ "$GPUS" -eq 1 ]; then
    python tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR"
# Multi-GPU training
else
    torchrun --nproc_per_node="$GPUS" \
        tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --launcher pytorch
fi

echo ""
echo "Optimized training complete!"
echo "Checkpoints saved to: $WORK_DIR"
echo "Best checkpoint: $WORK_DIR/best_coco_bbox_mAP_epoch_*.pth"
