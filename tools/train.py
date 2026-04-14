"""
train.py — Training entry point for MMDetection models.
Compatible with CUDA, MPS (Apple Silicon), and CPU.

Usage:
    # Mac (auto-detects MPS or CPU)
    python3 tools/train.py configs/baseline_faster_rcnn_r50_fpn_mac.py --work-dir work_dirs/baseline

    # CUDA (single GPU)
    python tools/train.py configs/baseline_faster_rcnn_r50_fpn.py --work-dir work_dirs/baseline

    # CUDA (multi-GPU)
    torchrun --nproc_per_node=4 tools/train.py configs/optimized_faster_rcnn_r101_pafpn.py \
        --work-dir work_dirs/optimized --launcher pytorch
"""

import argparse
import os
import os.path as osp
import platform
import sys

import torch
from mmengine.config import Config
from mmengine.runner import Runner


def detect_device():
    """Detect the best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def patch_config_for_device(cfg, device):
    """Patch config for non-CUDA devices (MPS/CPU)."""
    if device in ('mps', 'cpu'):
        # Force single-process data loading on macOS
        if hasattr(cfg, 'train_dataloader'):
            cfg.train_dataloader.num_workers = 0
            cfg.train_dataloader.persistent_workers = False
        if hasattr(cfg, 'val_dataloader'):
            cfg.val_dataloader.num_workers = 0
            cfg.val_dataloader.persistent_workers = False
        if hasattr(cfg, 'test_dataloader'):
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False

        # Use gloo backend instead of nccl
        if hasattr(cfg, 'env_cfg'):
            cfg.env_cfg.dist_cfg = dict(backend='gloo')
            cfg.env_cfg.mp_cfg = dict(mp_start_method='spawn', opencv_num_threads=0)

        # Disable AMP on MPS (not fully supported)
        if hasattr(cfg, 'optim_wrapper'):
            if isinstance(cfg.optim_wrapper, dict):
                if cfg.optim_wrapper.get('type') == 'AmpOptimWrapper':
                    cfg.optim_wrapper['type'] = 'OptimWrapper'
                    if 'loss_scale' in cfg.optim_wrapper:
                        del cfg.optim_wrapper['loss_scale']

        print(f"  Config patched for {device.upper()} device")


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='Directory to save logs and models')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Resume from the latest checkpoint'
    )
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Device to use (default: auto-detect)'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Detect device
    if args.device == 'auto':
        device = detect_device()
    else:
        device = args.device

    print(f"\n  Platform: {platform.system()} {platform.machine()}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  Device:   {device}")
    print()

    # Load config
    cfg = Config.fromfile(args.config)

    # Patch for Mac/CPU if needed
    patch_config_for_device(cfg, device)

    # Set launcher
    cfg.launcher = args.launcher

    # Set work_dir
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            'work_dirs',
            osp.splitext(osp.basename(args.config))[0]
        )

    # Set random seed
    cfg.seed = args.seed

    # Resume
    if args.resume:
        cfg.resume = True

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start training
    runner.train()


if __name__ == '__main__':
    main()
