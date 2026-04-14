"""
visualize_results.py — Visualize detection results and training curves.

Usage:
    python tools/visualize_results.py --mode curves \
        --baseline-log work_dirs/baseline_r50_fpn_1x/ \
        --optimized-log work_dirs/optimized_r101_pafpn_2x/

    python tools/visualize_results.py --mode detections \
        --config configs/optimized_faster_rcnn_r101_pafpn.py \
        --checkpoint work_dirs/optimized_r101_pafpn_2x/best.pth \
        --images data/coco/val2017/ \
        --output-dir results/visualizations/
"""

import argparse
import os
import json
import glob
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


def parse_training_log(work_dir):
    """Parse MMDetection JSON log files for loss and mAP curves."""
    log_files = sorted(glob.glob(os.path.join(work_dir, '*.log.json')))
    if not log_files:
        # Try MMEngine format
        log_files = sorted(glob.glob(os.path.join(work_dir, '**/scalars.json'), recursive=True))

    train_losses = []
    val_maps = []

    for log_file in log_files:
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if 'loss' in entry and 'mode' in entry and entry['mode'] == 'train':
                    train_losses.append({
                        'epoch': entry.get('epoch', 0),
                        'iter': entry.get('iter', 0),
                        'loss': entry['loss']
                    })
                elif 'coco/bbox_mAP' in entry:
                    val_maps.append({
                        'epoch': entry.get('epoch', entry.get('step', 0)),
                        'mAP': entry['coco/bbox_mAP']
                    })

    return train_losses, val_maps


def plot_training_curves(baseline_dir, optimized_dir, output_path='results/training_curves.png'):
    """Plot training loss and validation mAP curves for both models."""
    if not HAS_MPL:
        print("Cannot plot: matplotlib not installed")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, work_dir, color in [
        ('Baseline (R50-FPN)', baseline_dir, '#2196F3'),
        ('Optimized (R101-PAFPN)', optimized_dir, '#4CAF50')
    ]:
        train_losses, val_maps = parse_training_log(work_dir)

        if train_losses:
            epochs = [e['epoch'] + e['iter'] / 1000 for e in train_losses]
            losses = [e['loss'] for e in train_losses]
            # Smooth with moving average
            window = min(50, len(losses) // 5) if len(losses) > 10 else 1
            if window > 1:
                losses_smooth = np.convolve(losses, np.ones(window)/window, mode='valid')
                epochs_smooth = epochs[window-1:]
            else:
                losses_smooth = losses
                epochs_smooth = epochs
            ax1.plot(epochs_smooth, losses_smooth, label=label, color=color, alpha=0.8)

        if val_maps:
            epochs = [e['epoch'] for e in val_maps]
            maps = [e['mAP'] * 100 for e in val_maps]
            ax2.plot(epochs, maps, 'o-', label=label, color=color, markersize=4)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('mAP (%)')
    ax2.set_title('Validation mAP (COCO)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {output_path}")
    plt.close()


def visualize_detections(config, checkpoint, image_dir, output_dir, num_images=10):
    """Run inference and draw bounding boxes on sample images."""
    try:
        from mmdet.apis import init_detector, inference_detector
        from mmdet.registry import VISUALIZERS
        import mmcv
    except ImportError:
        print("MMDetection not installed. Install with: mim install mmdet")
        return

    # Build model
    model = init_detector(config, checkpoint, device='cuda:0')
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.dataset_meta = model.dataset_meta

    os.makedirs(output_dir, exist_ok=True)

    # Get image list
    image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:num_images]

    for img_path in image_files:
        img = mmcv.imread(img_path, channel_order='rgb')
        result = inference_detector(model, img)

        # Visualize
        visualizer.add_datasample(
            name=os.path.basename(img_path),
            image=img,
            data_sample=result,
            draw_gt=False,
            show=False,
            wait_time=0,
            out_file=os.path.join(output_dir, os.path.basename(img_path)),
            pred_score_thr=0.3
        )

    print(f"Visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize detection results')
    parser.add_argument(
        '--mode',
        choices=['curves', 'detections'],
        required=True,
        help='Visualization mode'
    )
    parser.add_argument('--baseline-log', default='work_dirs/baseline_r50_fpn_1x/')
    parser.add_argument('--optimized-log', default='work_dirs/optimized_r101_pafpn_2x/')
    parser.add_argument('--config', help='Config file for detection visualization')
    parser.add_argument('--checkpoint', help='Checkpoint for detection visualization')
    parser.add_argument('--images', help='Image directory')
    parser.add_argument('--output-dir', default='results/visualizations/')
    parser.add_argument('--num-images', type=int, default=10)
    args = parser.parse_args()

    if args.mode == 'curves':
        plot_training_curves(args.baseline_log, args.optimized_log)
    elif args.mode == 'detections':
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint are required for detection mode")
        visualize_detections(
            args.config, args.checkpoint,
            args.images or 'data/coco/val2017/',
            args.output_dir,
            args.num_images
        )


if __name__ == '__main__':
    main()
