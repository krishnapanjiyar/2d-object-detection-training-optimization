"""
test.py — Evaluation entry point for MMDetection models.
Compatible with CUDA, MPS (Apple Silicon), and CPU.

Usage:
    python3 tools/test.py configs/baseline_faster_rcnn_r50_fpn_mac.py \
        work_dirs/baseline/epoch_4.pth
"""

import argparse
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
        if hasattr(cfg, 'train_dataloader'):
            cfg.train_dataloader.num_workers = 0
            cfg.train_dataloader.persistent_workers = False
        if hasattr(cfg, 'val_dataloader'):
            cfg.val_dataloader.num_workers = 0
            cfg.val_dataloader.persistent_workers = False
        if hasattr(cfg, 'test_dataloader'):
            cfg.test_dataloader.num_workers = 0
            cfg.test_dataloader.persistent_workers = False
        if hasattr(cfg, 'env_cfg'):
            cfg.env_cfg.dist_cfg = dict(backend='gloo')
            cfg.env_cfg.mp_cfg = dict(mp_start_method='spawn', opencv_num_threads=0)
        if hasattr(cfg, 'optim_wrapper') and isinstance(cfg.optim_wrapper, dict):
            if cfg.optim_wrapper.get('type') == 'AmpOptimWrapper':
                cfg.optim_wrapper['type'] = 'OptimWrapper'
                cfg.optim_wrapper.pop('loss_scale', None)


def parse_args():
    parser = argparse.ArgumentParser(description='Test (evaluate) a detector')
    parser.add_argument('config', help='Test config file path')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out', help='Output result file in pickle format')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher'
    )
    parser.add_argument('--show', action='store_true', help='Show detection results')
    parser.add_argument('--show-dir', help='Directory to save visualization results')
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
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  Device:   {device}")
    print()

    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher

    # Patch for Mac/CPU
    patch_config_for_device(cfg, device)

    # Set work_dir for logs
    if cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            'work_dirs',
            osp.splitext(osp.basename(args.config))[0]
        )

    cfg.load_from = args.checkpoint

    # Handle visualization
    if args.show or args.show_dir:
        cfg.default_hooks.visualization.update(dict(draw=True, show=args.show))
        if args.show_dir:
            cfg.default_hooks.visualization.update(dict(test_out_dir=args.show_dir))

    # Build the runner
    runner = Runner.from_cfg(cfg)

    # Start testing
    runner.test()


if __name__ == '__main__':
    main()
