#!/bin/bash
# =============================================================
# Train Baseline: Faster R-CNN with ResNet-50-FPN (1x schedule)
# =============================================================

set -e

CONFIG="configs/baseline_faster_rcnn_r50_fpn.py"
WORK_DIR="work_dirs/baseline_r50_fpn_1x"
GPUS=${GPUS:-1}

echo "============================================"
echo "  Training Baseline Model"
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
echo "Baseline training complete!"
echo "Checkpoints saved to: $WORK_DIR"
echo "Best checkpoint: $WORK_DIR/best_coco_bbox_mAP_epoch_*.pth"
