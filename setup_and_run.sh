#!/bin/bash
# =============================================================
# setup_and_run.sh — Complete end-to-end pipeline
# Downloads COCO, installs dependencies, trains both models,
# evaluates, and generates comparison.
#
# Usage:
#   bash setup_and_run.sh [--skip-download] [--gpus N]
#
# Requirements:
#   - NVIDIA GPU with >= 12GB VRAM (24GB recommended)
#   - ~25GB free disk space for COCO dataset
#   - Python 3.8+ with pip
#   - CUDA 11.7+ installed
# =============================================================

set -e

SKIP_DOWNLOAD=false
GPUS=1
DATA_ROOT="data/coco"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --gpus) GPUS=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  2D Object Detection Training Optimization"
echo "  End-to-End Pipeline"
echo "============================================"
echo ""

# ----------------------------------------------------------
# Step 1: Install Dependencies
# ----------------------------------------------------------
echo "[1/7] Installing dependencies..."

pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -q -U openmim
mim install -q mmengine mmcv mmdet
pip install -q pycocotools matplotlib seaborn

echo "  ✓ Dependencies installed"
echo ""

# ----------------------------------------------------------
# Step 2: Download COCO Dataset
# ----------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[2/7] Downloading COCO 2017 dataset..."
    mkdir -p "$DATA_ROOT"
    cd "$DATA_ROOT"

    # Train images (~18GB)
    if [ ! -d "train2017" ]; then
        echo "  Downloading train2017 images..."
        wget -q --show-progress http://images.cocodataset.org/zips/train2017.zip
        unzip -q train2017.zip && rm train2017.zip
    else
        echo "  train2017 already exists, skipping."
    fi

    # Val images (~1GB)
    if [ ! -d "val2017" ]; then
        echo "  Downloading val2017 images..."
        wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip
        unzip -q val2017.zip && rm val2017.zip
    else
        echo "  val2017 already exists, skipping."
    fi

    # Annotations (~241MB)
    if [ ! -d "annotations" ]; then
        echo "  Downloading annotations..."
        wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q annotations_trainval2017.zip && rm annotations_trainval2017.zip
    else
        echo "  annotations already exists, skipping."
    fi

    cd - > /dev/null
    echo "  ✓ COCO 2017 dataset ready at $DATA_ROOT"
else
    echo "[2/7] Skipping COCO download (--skip-download)"
fi
echo ""

# ----------------------------------------------------------
# Step 3: Verify dataset structure
# ----------------------------------------------------------
echo "[3/7] Verifying dataset structure..."

EXPECTED_DIRS=("$DATA_ROOT/train2017" "$DATA_ROOT/val2017" "$DATA_ROOT/annotations")
for dir in "${EXPECTED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "  ✗ Missing: $dir"
        echo "  Please download COCO 2017 or run without --skip-download"
        exit 1
    fi
done

TRAIN_COUNT=$(ls "$DATA_ROOT/train2017/"*.jpg 2>/dev/null | wc -l)
VAL_COUNT=$(ls "$DATA_ROOT/val2017/"*.jpg 2>/dev/null | wc -l)
echo "  Train images: $TRAIN_COUNT"
echo "  Val images:   $VAL_COUNT"
echo "  ✓ Dataset structure verified"
echo ""

# ----------------------------------------------------------
# Step 4: Train Baseline
# ----------------------------------------------------------
echo "[4/7] Training BASELINE model (Faster R-CNN R50-FPN, 12 epochs)..."
echo "  This may take several hours depending on your GPU."

BASELINE_WORK_DIR="work_dirs/baseline_r50_fpn_1x"
mkdir -p "$BASELINE_WORK_DIR"

if [ "$GPUS" -eq 1 ]; then
    python tools/train.py \
        configs/baseline_faster_rcnn_r50_fpn.py \
        --work-dir "$BASELINE_WORK_DIR" \
        2>&1 | tee "$BASELINE_WORK_DIR/train.log"
else
    torchrun --nproc_per_node="$GPUS" \
        tools/train.py \
        configs/baseline_faster_rcnn_r50_fpn.py \
        --work-dir "$BASELINE_WORK_DIR" \
        --launcher pytorch \
        2>&1 | tee "$BASELINE_WORK_DIR/train.log"
fi

echo "  ✓ Baseline training complete"
echo ""

# ----------------------------------------------------------
# Step 5: Train Optimized
# ----------------------------------------------------------
echo "[5/7] Training OPTIMIZED model (Faster R-CNN R101-PAFPN, 24 epochs)..."
echo "  This will take longer than baseline."

OPTIMIZED_WORK_DIR="work_dirs/optimized_r101_pafpn_2x"
mkdir -p "$OPTIMIZED_WORK_DIR"

if [ "$GPUS" -eq 1 ]; then
    python tools/train.py \
        configs/optimized_faster_rcnn_r101_pafpn.py \
        --work-dir "$OPTIMIZED_WORK_DIR" \
        2>&1 | tee "$OPTIMIZED_WORK_DIR/train.log"
else
    torchrun --nproc_per_node="$GPUS" \
        tools/train.py \
        configs/optimized_faster_rcnn_r101_pafpn.py \
        --work-dir "$OPTIMIZED_WORK_DIR" \
        --launcher pytorch \
        2>&1 | tee "$OPTIMIZED_WORK_DIR/train.log"
fi

echo "  ✓ Optimized training complete"
echo ""

# ----------------------------------------------------------
# Step 6: Evaluate Both Models
# ----------------------------------------------------------
echo "[6/7] Evaluating both models on COCO val2017..."
mkdir -p results

# Find best checkpoints
BASELINE_CKPT=$(ls -t "$BASELINE_WORK_DIR"/best_*.pth 2>/dev/null | head -1)
if [ -z "$BASELINE_CKPT" ]; then
    BASELINE_CKPT=$(ls -t "$BASELINE_WORK_DIR"/epoch_12.pth 2>/dev/null | head -1)
fi

OPTIMIZED_CKPT=$(ls -t "$OPTIMIZED_WORK_DIR"/best_*.pth 2>/dev/null | head -1)
if [ -z "$OPTIMIZED_CKPT" ]; then
    OPTIMIZED_CKPT=$(ls -t "$OPTIMIZED_WORK_DIR"/epoch_24.pth 2>/dev/null | head -1)
fi

echo "  Baseline checkpoint:  $BASELINE_CKPT"
echo "  Optimized checkpoint: $OPTIMIZED_CKPT"

echo "  Evaluating baseline..."
python tools/test.py \
    configs/baseline_faster_rcnn_r50_fpn.py \
    "$BASELINE_CKPT" \
    2>&1 | tee results/baseline_eval.log

echo "  Evaluating optimized..."
python tools/test.py \
    configs/optimized_faster_rcnn_r101_pafpn.py \
    "$OPTIMIZED_CKPT" \
    2>&1 | tee results/optimized_eval.log

echo "  ✓ Evaluation complete"
echo ""

# ----------------------------------------------------------
# Step 7: Generate Comparison & Visualizations
# ----------------------------------------------------------
echo "[7/7] Generating comparison tables and visualizations..."

python scripts/compare_results.py \
    --baseline-log results/baseline_eval.log \
    --optimized-log results/optimized_eval.log \
    --output-dir results

python tools/visualize_results.py \
    --mode curves \
    --baseline-log "$BASELINE_WORK_DIR" \
    --optimized-log "$OPTIMIZED_WORK_DIR"

python tools/visualize_results.py \
    --mode detections \
    --config configs/optimized_faster_rcnn_r101_pafpn.py \
    --checkpoint "$OPTIMIZED_CKPT" \
    --images "$DATA_ROOT/val2017/" \
    --output-dir results/visualizations/ \
    --num-images 20

echo "  ✓ Results generated"
echo ""

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo ""
echo "  Results:"
echo "    results/comparison_table.md   — Side-by-side mAP comparison"
echo "    results/results.json          — Machine-readable results"
echo "    results/baseline_eval.log     — Full baseline eval output"
echo "    results/optimized_eval.log    — Full optimized eval output"
echo "    results/training_curves.png   — Loss & mAP plots"
echo "    results/visualizations/       — Detection visualizations"
echo ""
echo "  Checkpoints:"
echo "    $BASELINE_CKPT"
echo "    $OPTIMIZED_CKPT"
echo ""
echo "  Custom annotations:"
echo "    custom_dataset/annotations/custom_annotations.json"
echo ""
cat results/comparison_table.md
