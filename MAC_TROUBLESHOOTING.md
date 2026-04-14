# macOS Troubleshooting Guide

## Common Issues and Fixes

### 1. "No module named 'mmcv'" or mmdet/mmengine errors

```bash
# Use mim installer (handles dependencies correctly)
pip3 install -U openmim
mim install mmengine "mmcv>=2.0.0" mmdet
```

If mim fails, install from source:
```bash
pip3 install mmengine
pip3 install mmcv==2.1.0
pip3 install mmdet
```

### 2. "RuntimeError: Placeholder storage has not been allocated on MPS device!"

MPS doesn't support all PyTorch operations. Fix:
```bash
# Set fallback to CPU for unsupported ops
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Or run with CPU-only
python3 tools/train.py configs/baseline_faster_rcnn_r50_fpn_mac.py --device cpu
```

### 3. "fork() crash" or multiprocessing errors

macOS requires `spawn` instead of `fork`. The Mac configs already handle this, but if using the original configs:
```python
# In your config, ensure:
env_cfg = dict(
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='gloo')
)

# And in dataloader:
train_dataloader = dict(
    num_workers=0,              # <-- Must be 0 on macOS
    persistent_workers=False,   # <-- Must be False
)
```

### 4. "libomp" / OpenMP errors

```bash
# Install OpenMP for macOS
brew install libomp

# If still failing, set:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

### 5. Out of Memory (OOM)

Mac unified memory is shared between CPU and GPU. Reduce memory usage:
```python
# In config, reduce batch size:
train_dataloader = dict(batch_size=1)

# Reduce image size:
train_pipeline = [
    dict(type='Resize', scale=(800, 600), keep_ratio=True),  # Smaller
]
```

### 6. pycocotools installation fails

```bash
# Install build dependencies first
brew install gcc
pip3 install cython numpy
pip3 install pycocotools
```

### 7. Training is very slow

Expected — Mac training is 5-10x slower than a CUDA GPU. Tips:
- Use the `_mac.py` configs which have reduced epochs (4 baseline, 8 optimized)
- Use `--mini` flag for evaluation only (skip full training)
- Consider training on Google Colab (free CUDA GPU) and just evaluating locally
- If you have Apple M2 Pro/Max/Ultra, it will be faster than M1

### 8. NCCL backend error

The Mac configs use `gloo` backend. If you see NCCL errors, ensure you're using the `_mac.py` configs:
```bash
python3 tools/train.py configs/baseline_faster_rcnn_r50_fpn_mac.py  # ✓ correct
python3 tools/train.py configs/baseline_faster_rcnn_r50_fpn.py       # ✗ uses NCCL
```

### 9. cv2 (OpenCV) import error

```bash
pip3 install opencv-python-headless
# or
pip3 install opencv-python
```

### 10. How to check if MPS is working

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Quick test
if torch.backends.mps.is_available():
    x = torch.randn(3, 3, device='mps')
    print(f"Tensor on MPS: {x.device}")  # Should show 'mps:0'
```

---

## Recommended Workflow for Mac

Since Mac training is slow, here's the most practical approach:

1. **Generate configs**: `python3 scripts/generate_mac_configs.py --device mps`
2. **Train baseline** (4 epochs): ~2 hours on M1
3. **Train optimized** (8 epochs): ~5 hours on M1
4. **Evaluate both** and generate comparison
5. **Annotate custom data** with Label Studio (runs great on Mac)
6. **Fine-tune** with custom data (6 more epochs)

Or for faster results, train on Google Colab and download checkpoints to Mac for evaluation.

## Google Colab Alternative

If Mac training is too slow:

```python
# In Colab (free GPU):
!pip install -U openmim
!mim install mmengine mmcv mmdet

# Upload configs, train, download checkpoints
# Then evaluate locally on your Mac
```
