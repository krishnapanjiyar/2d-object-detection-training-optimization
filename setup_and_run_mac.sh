#!/bin/bash
# =============================================================
# setup_and_run_mac.sh — macOS-compatible end-to-end pipeline
#
# Supports:
#   - Apple Silicon (M1/M2/M3/M4) via MPS backend
#   - Intel Mac via CPU
#
# Usage:
#   bash setup_and_run_mac.sh [--skip-download] [--cpu-only] [--mini]
#
# Options:
#   --skip-download   Skip COCO dataset download
#   --cpu-only        Force CPU even on Apple Silicon
#   --mini            Use COCO mini val (5K images) for faster eval
#
# Requirements:
#   - macOS 12.3+ (for MPS) or any macOS (CPU)
#   - Python 3.9+ (recommend 3.10 or 3.11)
#   - ~25GB disk for full COCO, or ~1GB for mini setup
#   - 16GB+ RAM recommended
# =============================================================

set -e

SKIP_DOWNLOAD=false
CPU_ONLY=false
MINI_MODE=false
DATA_ROOT="data/coco"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --cpu-only) CPU_ONLY=true; shift ;;
        --mini) MINI_MODE=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  2D Object Detection Training Optimization"
echo "  macOS Pipeline"
echo "============================================"

# ----------------------------------------------------------
# Detect hardware
# ----------------------------------------------------------
if [[ $(uname -m) == "arm64" ]] && [[ "$CPU_ONLY" == false ]]; then
    DEVICE="mps"
    echo "  Hardware: Apple Silicon detected → using MPS"
else
    DEVICE="cpu"
    echo "  Hardware: Using CPU"
fi
echo ""

# ----------------------------------------------------------
# Step 1: Install Dependencies
# ----------------------------------------------------------
echo "[1/7] Installing dependencies..."

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
echo "  Python version: $PYTHON_VERSION"

# Python 3.9 from Apple CLT is too old (missing pkg_resources, setuptools issues)
# Need Python 3.10+ for mmcv to build properly
if [ "$PYTHON_MINOR" -lt 10 ]; then
    echo ""
    echo "  ⚠ Python $PYTHON_VERSION is too old for mmcv. Need Python 3.10+."
    echo ""

    # Check if Homebrew Python 3.10+ exists
    if command -v /opt/homebrew/bin/python3 &>/dev/null; then
        BREW_PY_VER=$(/opt/homebrew/bin/python3 -c "import sys; print(sys.version_info.minor)")
        if [ "$BREW_PY_VER" -ge 10 ]; then
            echo "  Found Homebrew Python 3.$BREW_PY_VER — using that instead."
            alias python3="/opt/homebrew/bin/python3"
            alias pip3="/opt/homebrew/bin/pip3"
            PYTHON_VERSION="3.$BREW_PY_VER"
        fi
    fi

    # Re-check after potential alias
    PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "9")
    if [ "$PYTHON_MINOR" -lt 10 ]; then
        echo ""
        echo "  ✗ No Python 3.10+ found. Please install it first:"
        echo ""
        echo "    Option A (Homebrew — recommended):"
        echo "      brew install python@3.11"
        echo "      echo 'export PATH=\"/opt/homebrew/bin:\$PATH\"' >> ~/.zshrc"
        echo "      source ~/.zshrc"
        echo ""
        echo "    Option B (python.org installer):"
        echo "      Download from https://www.python.org/downloads/"
        echo ""
        echo "  Then re-run: bash setup_and_run_mac.sh"
        exit 1
    fi
fi

# Add user bin to PATH (pip installs scripts here on macOS)
export PATH="$HOME/Library/Python/$PYTHON_VERSION/bin:$HOME/.local/bin:/opt/homebrew/bin:$PATH"

# Upgrade pip and install setuptools (needed for mmcv build)
python3 -m pip install -q --upgrade pip setuptools wheel

# Install PyTorch (MPS-compatible version for macOS)
python3 -m pip install -q torch torchvision

# Install MMDetection ecosystem
python3 -m pip install -q -U openmim
python3 -m mim install -q mmengine
python3 -m pip install -q "mmcv>=2.0.0,<2.2.0"
python3 -m pip install -q "mmdet>=3.2.0,<3.4.0"

# Additional dependencies
python3 -m pip install -q pycocotools matplotlib seaborn Pillow opencv-python

# Verify MPS availability
if [[ "$DEVICE" == "mps" ]]; then
    MPS_AVAIL=$(python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "False")
    if [[ "$MPS_AVAIL" != "True" ]]; then
        echo "  ⚠ MPS not available, falling back to CPU"
        DEVICE="cpu"
    else
        echo "  ✓ MPS backend verified"
    fi
fi

echo "  ✓ Dependencies installed (device=$DEVICE)"
echo ""

# ----------------------------------------------------------
# Step 2: Download COCO Dataset
# ----------------------------------------------------------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "[2/7] Downloading COCO 2017 dataset..."
    mkdir -p "$DATA_ROOT"

    # Val images (~1GB) — always needed
    if [ ! -d "$DATA_ROOT/val2017" ]; then
        echo "  Downloading val2017 images (~1GB)..."
        curl -L -# -o "$DATA_ROOT/val2017.zip" \
            http://images.cocodataset.org/zips/val2017.zip
        unzip -q "$DATA_ROOT/val2017.zip" -d "$DATA_ROOT" && rm "$DATA_ROOT/val2017.zip"
    else
        echo "  val2017 already exists, skipping."
    fi

    # Annotations (~241MB)
    if [ ! -d "$DATA_ROOT/annotations" ]; then
        echo "  Downloading annotations (~241MB)..."
        curl -L -# -o "$DATA_ROOT/annotations.zip" \
            http://images.cocodataset.org/annotations/annotations_trainval2017.zip
        unzip -q "$DATA_ROOT/annotations.zip" -d "$DATA_ROOT" && rm "$DATA_ROOT/annotations.zip"
    else
        echo "  annotations already exists, skipping."
    fi

    # Train images (~18GB) — skip in mini mode
    if [ "$MINI_MODE" = false ]; then
        if [ ! -d "$DATA_ROOT/train2017" ]; then
            echo "  Downloading train2017 images (~18GB, this will take a while)..."
            curl -L -# -o "$DATA_ROOT/train2017.zip" \
                http://images.cocodataset.org/zips/train2017.zip
            unzip -q "$DATA_ROOT/train2017.zip" -d "$DATA_ROOT" && rm "$DATA_ROOT/train2017.zip"
        else
            echo "  train2017 already exists, skipping."
        fi
    else
        echo "  Mini mode: skipping train2017 download (eval-only)"
    fi

    echo "  ✓ COCO dataset ready"
else
    echo "[2/7] Skipping COCO download (--skip-download)"
fi
echo ""

# ----------------------------------------------------------
# Step 3: Generate Mac-compatible configs
# ----------------------------------------------------------
echo "[3/7] Generating macOS-compatible configs..."

python3 scripts/generate_mac_configs.py --device "$DEVICE"

echo "  ✓ Configs generated for device=$DEVICE"
echo ""

# ----------------------------------------------------------
# Step 4: Train Baseline (reduced schedule for Mac)
# ----------------------------------------------------------
echo "[4/7] Training BASELINE model..."
echo "  ⚠ Training on Mac is slow. Consider using --mini for eval-only."
echo "  Estimated time: ~2-4 hours on M1/M2 (reduced 4-epoch schedule)"

BASELINE_WORK_DIR="work_dirs/baseline_r50_fpn_mac"
mkdir -p "$BASELINE_WORK_DIR"

python3 tools/train.py \
    configs/baseline_faster_rcnn_r50_fpn_mac.py \
    --work-dir "$BASELINE_WORK_DIR" \
    2>&1 | tee "$BASELINE_WORK_DIR/train.log"

echo "  ✓ Baseline training complete"
echo ""

# ----------------------------------------------------------
# Step 5: Train Optimized (reduced schedule for Mac)
# ----------------------------------------------------------
echo "[5/7] Training OPTIMIZED model..."
echo "  Estimated time: ~4-8 hours on M1/M2 (reduced 8-epoch schedule)"

OPTIMIZED_WORK_DIR="work_dirs/optimized_r101_pafpn_mac"
mkdir -p "$OPTIMIZED_WORK_DIR"

python3 tools/train.py \
    configs/optimized_faster_rcnn_r101_pafpn_mac.py \
    --work-dir "$OPTIMIZED_WORK_DIR" \
    2>&1 | tee "$OPTIMIZED_WORK_DIR/train.log"

echo "  ✓ Optimized training complete"
echo ""

# ----------------------------------------------------------
# Step 6: Evaluate
# ----------------------------------------------------------
echo "[6/7] Evaluating both models..."
mkdir -p results

BASELINE_CKPT=$(ls -t "$BASELINE_WORK_DIR"/best_*.pth "$BASELINE_WORK_DIR"/epoch_*.pth 2>/dev/null | head -1)
OPTIMIZED_CKPT=$(ls -t "$OPTIMIZED_WORK_DIR"/best_*.pth "$OPTIMIZED_WORK_DIR"/epoch_*.pth 2>/dev/null | head -1)

echo "  Baseline checkpoint:  $BASELINE_CKPT"
echo "  Optimized checkpoint: $OPTIMIZED_CKPT"

if [ -n "$BASELINE_CKPT" ]; then
    python3 tools/test.py \
        configs/baseline_faster_rcnn_r50_fpn_mac.py \
        "$BASELINE_CKPT" \
        2>&1 | tee results/baseline_eval.log
fi

if [ -n "$OPTIMIZED_CKPT" ]; then
    python3 tools/test.py \
        configs/optimized_faster_rcnn_r101_pafpn_mac.py \
        "$OPTIMIZED_CKPT" \
        2>&1 | tee results/optimized_eval.log
fi

echo "  ✓ Evaluation complete"
echo ""

# ----------------------------------------------------------
# Step 7: Generate comparison
# ----------------------------------------------------------
echo "[7/7] Generating comparison..."

python3 scripts/compare_results.py \
    --baseline-log results/baseline_eval.log \
    --optimized-log results/optimized_eval.log \
    --output-dir results

echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo ""
echo "  Results:"
echo "    results/comparison_table.md"
echo "    results/results.json"
echo "    results/baseline_eval.log"
echo "    results/optimized_eval.log"
echo ""
echo "  Next steps:"
echo "    1. Add your own images to custom_dataset/images/"
echo "    2. Annotate them (see custom_dataset/ANNOTATION_GUIDE.md)"
echo "    3. Fine-tune: python3 tools/train.py configs/finetune_with_custom_data.py"
