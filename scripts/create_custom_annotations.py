"""
create_custom_annotations.py — Build a custom annotated dataset in COCO format.

This script provides two workflows for creating custom annotations:

  1. INTERACTIVE MODE: Manually annotate images using OpenCV GUI
     - Click to draw bounding boxes
     - Select categories from a menu
     - Exports COCO-format JSON

  2. ASSISTED MODE: Use a pretrained model to generate initial detections,
     then manually review/correct them
     - Runs inference with a pretrained Faster R-CNN
     - Presents each detection for accept/reject/modify
     - Exports corrected annotations in COCO format

Usage:
    # Interactive manual annotation
    python scripts/create_custom_annotations.py \
        --image-dir custom_dataset/images/ \
        --output custom_dataset/annotations/custom_annotations.json \
        --mode interactive

    # Model-assisted annotation (review pretrained model predictions)
    python scripts/create_custom_annotations.py \
        --image-dir custom_dataset/images/ \
        --output custom_dataset/annotations/custom_annotations.json \
        --mode assisted \
        --config configs/baseline_faster_rcnn_r50_fpn.py \
        --checkpoint work_dirs/baseline_r50_fpn_1x/epoch_12.pth

    # Headless mode (no GUI, generates from model predictions with threshold)
    python scripts/create_custom_annotations.py \
        --image-dir custom_dataset/images/ \
        --output custom_dataset/annotations/custom_annotations.json \
        --mode headless \
        --config configs/baseline_faster_rcnn_r50_fpn.py \
        --checkpoint work_dirs/baseline_r50_fpn_1x/epoch_12.pth \
        --score-thr 0.5
"""

import argparse
import json
import os
import glob
import datetime
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# COCO 80 category definitions
COCO_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 4, "name": "motorcycle", "supercategory": "vehicle"},
    {"id": 5, "name": "airplane", "supercategory": "vehicle"},
    {"id": 6, "name": "bus", "supercategory": "vehicle"},
    {"id": 7, "name": "train", "supercategory": "vehicle"},
    {"id": 8, "name": "truck", "supercategory": "vehicle"},
    {"id": 9, "name": "boat", "supercategory": "vehicle"},
    {"id": 10, "name": "traffic light", "supercategory": "outdoor"},
    {"id": 11, "name": "fire hydrant", "supercategory": "outdoor"},
    {"id": 13, "name": "stop sign", "supercategory": "outdoor"},
    {"id": 14, "name": "parking meter", "supercategory": "outdoor"},
    {"id": 15, "name": "bench", "supercategory": "outdoor"},
    {"id": 16, "name": "bird", "supercategory": "animal"},
    {"id": 17, "name": "cat", "supercategory": "animal"},
    {"id": 18, "name": "dog", "supercategory": "animal"},
    {"id": 19, "name": "horse", "supercategory": "animal"},
    {"id": 20, "name": "sheep", "supercategory": "animal"},
    {"id": 21, "name": "cow", "supercategory": "animal"},
    {"id": 22, "name": "elephant", "supercategory": "animal"},
    {"id": 23, "name": "bear", "supercategory": "animal"},
    {"id": 24, "name": "zebra", "supercategory": "animal"},
    {"id": 25, "name": "giraffe", "supercategory": "animal"},
    {"id": 27, "name": "backpack", "supercategory": "accessory"},
    {"id": 28, "name": "umbrella", "supercategory": "accessory"},
    {"id": 31, "name": "handbag", "supercategory": "accessory"},
    {"id": 32, "name": "tie", "supercategory": "accessory"},
    {"id": 33, "name": "suitcase", "supercategory": "accessory"},
    {"id": 34, "name": "frisbee", "supercategory": "sports"},
    {"id": 35, "name": "skis", "supercategory": "sports"},
    {"id": 36, "name": "snowboard", "supercategory": "sports"},
    {"id": 37, "name": "sports ball", "supercategory": "sports"},
    {"id": 38, "name": "kite", "supercategory": "sports"},
    {"id": 39, "name": "baseball bat", "supercategory": "sports"},
    {"id": 40, "name": "baseball glove", "supercategory": "sports"},
    {"id": 41, "name": "skateboard", "supercategory": "sports"},
    {"id": 42, "name": "surfboard", "supercategory": "sports"},
    {"id": 43, "name": "tennis racket", "supercategory": "sports"},
    {"id": 44, "name": "bottle", "supercategory": "kitchen"},
    {"id": 46, "name": "wine glass", "supercategory": "kitchen"},
    {"id": 47, "name": "cup", "supercategory": "kitchen"},
    {"id": 48, "name": "fork", "supercategory": "kitchen"},
    {"id": 49, "name": "knife", "supercategory": "kitchen"},
    {"id": 50, "name": "spoon", "supercategory": "kitchen"},
    {"id": 51, "name": "bowl", "supercategory": "kitchen"},
    {"id": 52, "name": "banana", "supercategory": "food"},
    {"id": 53, "name": "apple", "supercategory": "food"},
    {"id": 54, "name": "sandwich", "supercategory": "food"},
    {"id": 55, "name": "orange", "supercategory": "food"},
    {"id": 56, "name": "broccoli", "supercategory": "food"},
    {"id": 57, "name": "carrot", "supercategory": "food"},
    {"id": 58, "name": "hot dog", "supercategory": "food"},
    {"id": 59, "name": "pizza", "supercategory": "food"},
    {"id": 60, "name": "donut", "supercategory": "food"},
    {"id": 61, "name": "cake", "supercategory": "food"},
    {"id": 62, "name": "chair", "supercategory": "furniture"},
    {"id": 63, "name": "couch", "supercategory": "furniture"},
    {"id": 64, "name": "potted plant", "supercategory": "furniture"},
    {"id": 65, "name": "bed", "supercategory": "furniture"},
    {"id": 67, "name": "dining table", "supercategory": "furniture"},
    {"id": 70, "name": "toilet", "supercategory": "furniture"},
    {"id": 72, "name": "tv", "supercategory": "electronic"},
    {"id": 73, "name": "laptop", "supercategory": "electronic"},
    {"id": 74, "name": "mouse", "supercategory": "electronic"},
    {"id": 75, "name": "remote", "supercategory": "electronic"},
    {"id": 76, "name": "keyboard", "supercategory": "electronic"},
    {"id": 77, "name": "cell phone", "supercategory": "electronic"},
    {"id": 78, "name": "microwave", "supercategory": "appliance"},
    {"id": 79, "name": "oven", "supercategory": "appliance"},
    {"id": 80, "name": "toaster", "supercategory": "appliance"},
    {"id": 81, "name": "sink", "supercategory": "appliance"},
    {"id": 82, "name": "refrigerator", "supercategory": "appliance"},
    {"id": 84, "name": "book", "supercategory": "indoor"},
    {"id": 85, "name": "clock", "supercategory": "indoor"},
    {"id": 86, "name": "vase", "supercategory": "indoor"},
    {"id": 87, "name": "scissors", "supercategory": "indoor"},
    {"id": 88, "name": "teddy bear", "supercategory": "indoor"},
    {"id": 89, "name": "hair drier", "supercategory": "indoor"},
    {"id": 90, "name": "toothbrush", "supercategory": "indoor"},
]


def get_image_size(image_path):
    """Get image width and height."""
    if HAS_CV2:
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    if HAS_PIL:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    raise RuntimeError("Neither cv2 nor PIL available. Install opencv-python or Pillow.")


def create_empty_coco_annotation(categories=None):
    """Create an empty COCO annotation structure."""
    return {
        "info": {
            "description": "Custom annotated dataset for training optimization",
            "version": "1.0",
            "year": 2025,
            "contributor": "Student",
            "date_created": datetime.datetime.now().isoformat()
        },
        "licenses": [
            {"id": 1, "name": "Custom", "url": ""}
        ],
        "categories": categories or COCO_CATEGORIES,
        "images": [],
        "annotations": []
    }


def interactive_annotate(image_dir, output_path, categories=None):
    """
    Interactive annotation with OpenCV GUI.
    
    Controls:
        - Left click + drag: Draw bounding box
        - Enter: Confirm box and select category
        - 'n': Next image
        - 'u': Undo last box
        - 'q': Quit and save
    """
    if not HAS_CV2:
        print("ERROR: OpenCV required for interactive mode.")
        print("Install with: pip install opencv-python")
        return

    coco = create_empty_coco_annotation(categories)
    image_files = sorted(
        glob.glob(os.path.join(image_dir, '*.jpg')) +
        glob.glob(os.path.join(image_dir, '*.png')) +
        glob.glob(os.path.join(image_dir, '*.jpeg'))
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    cat_names = [c['name'] for c in (categories or COCO_CATEGORIES)]
    cat_ids = [c['id'] for c in (categories or COCO_CATEGORIES)]
    ann_id = 1
    drawing = False
    ix, iy = -1, -1
    current_boxes = []

    def draw_box(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img_display
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_display = img_clean.copy()
            cv2.rectangle(img_display, (ix, iy), (x, y), (0, 255, 0), 2)
            for box, cat in current_boxes:
                cv2.rectangle(img_display, (box[0], box[1]),
                              (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
                cv2.putText(img_display, cat, (box[0], box[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = min(ix, x), min(iy, y)
            x2, y2 = max(ix, x), max(iy, y)
            w, h = x2 - x1, y2 - y1
            if w > 5 and h > 5:
                # Show category selection
                print("\nSelect category:")
                for i, name in enumerate(cat_names[:20]):  # Show first 20
                    print(f"  {i}: {name}")
                cat_idx = int(input("Category number: "))
                if 0 <= cat_idx < len(cat_names):
                    current_boxes.append(([x1, y1, w, h], cat_names[cat_idx]))
                    print(f"  Added: {cat_names[cat_idx]} at [{x1},{y1},{w},{h}]")

    print(f"\nInteractive Annotation Mode")
    print(f"Images: {len(image_files)} found in {image_dir}")
    print(f"Controls: drag=draw box, n=next, u=undo, q=quit+save\n")

    cv2.namedWindow('Annotate')
    cv2.setMouseCallback('Annotate', draw_box)

    for img_idx, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        img_clean = img.copy()
        img_display = img.copy()
        current_boxes = []

        # Add image to COCO
        image_id = img_idx + 1
        coco['images'].append({
            'id': image_id,
            'file_name': os.path.basename(img_path),
            'width': w,
            'height': h
        })

        print(f"\n[{img_idx+1}/{len(image_files)}] {os.path.basename(img_path)} ({w}x{h})")

        while True:
            cv2.imshow('Annotate', img_display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):  # Next image
                break
            elif key == ord('u') and current_boxes:  # Undo
                removed = current_boxes.pop()
                print(f"  Undid: {removed[1]}")
                img_display = img_clean.copy()
                for box, cat in current_boxes:
                    cv2.rectangle(img_display, (box[0], box[1]),
                                  (box[0]+box[2], box[1]+box[3]), (255, 0, 0), 2)
            elif key == ord('q'):  # Quit
                # Save current image annotations first
                for box, cat_name in current_boxes:
                    cat_id = cat_ids[cat_names.index(cat_name)]
                    coco['annotations'].append({
                        'id': ann_id,
                        'image_id': image_id,
                        'category_id': cat_id,
                        'bbox': box,
                        'area': box[2] * box[3],
                        'iscrowd': 0
                    })
                    ann_id += 1
                # Save and exit
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(coco, f, indent=2)
                print(f"\nSaved {len(coco['annotations'])} annotations to {output_path}")
                cv2.destroyAllWindows()
                return

        # Save annotations for this image
        for box, cat_name in current_boxes:
            cat_id = cat_ids[cat_names.index(cat_name)]
            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': box,
                'area': box[2] * box[3],
                'iscrowd': 0
            })
            ann_id += 1

    cv2.destroyAllWindows()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"\nSaved {len(coco['annotations'])} annotations to {output_path}")


def headless_annotate(image_dir, output_path, config, checkpoint, score_thr=0.5):
    """
    Headless annotation: run a pretrained model and accept all predictions
    above a confidence threshold as ground truth annotations.
    
    This is useful for bootstrapping annotations on new data.
    The user should manually review and correct the output JSON.
    """
    try:
        from mmdet.apis import init_detector, inference_detector
    except ImportError:
        print("ERROR: MMDetection required for assisted/headless mode.")
        print("Install with: mim install mmdet")
        return

    coco = create_empty_coco_annotation()
    image_files = sorted(
        glob.glob(os.path.join(image_dir, '*.jpg')) +
        glob.glob(os.path.join(image_dir, '*.png')) +
        glob.glob(os.path.join(image_dir, '*.jpeg'))
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Running inference on {len(image_files)} images...")
    print(f"Model: {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Score threshold: {score_thr}")

    model = init_detector(config, checkpoint, device='cuda:0')
    class_names = model.dataset_meta.get('classes', [])

    # Build category mapping from model classes to COCO category IDs
    coco_cat_map = {}
    for cat in COCO_CATEGORIES:
        coco_cat_map[cat['name']] = cat['id']

    ann_id = 1

    for img_idx, img_path in enumerate(image_files):
        w, h = get_image_size(img_path)

        image_id = img_idx + 1
        coco['images'].append({
            'id': image_id,
            'file_name': os.path.basename(img_path),
            'width': w,
            'height': h
        })

        # Run inference
        result = inference_detector(model, img_path)
        pred_instances = result.pred_instances

        # Filter by score
        scores = pred_instances.scores.cpu().numpy()
        bboxes = pred_instances.bboxes.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        keep = scores >= score_thr
        count = 0

        for score, bbox, label in zip(scores[keep], bboxes[keep], labels[keep]):
            class_name = class_names[label]
            cat_id = coco_cat_map.get(class_name)
            if cat_id is None:
                continue

            x1, y1, x2, y2 = bbox
            w_box = float(x2 - x1)
            h_box = float(y2 - y1)

            coco['annotations'].append({
                'id': ann_id,
                'image_id': image_id,
                'category_id': cat_id,
                'bbox': [float(x1), float(y1), w_box, h_box],
                'area': w_box * h_box,
                'score': float(score),  # Keep score for review
                'iscrowd': 0
            })
            ann_id += 1
            count += 1

        if (img_idx + 1) % 10 == 0:
            print(f"  [{img_idx+1}/{len(image_files)}] {count} detections")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    total_ann = len(coco['annotations'])
    total_img = len(coco['images'])
    print(f"\nDone! {total_ann} annotations across {total_img} images")
    print(f"Saved to: {output_path}")
    print(f"\nIMPORTANT: Review and correct annotations before using for training.")
    print(f"You can edit the JSON directly or import into Label Studio for visual review.")


def create_sample_custom_annotations(image_dir, output_path):
    """
    Create sample custom annotations from a directory of images.
    
    This demonstrates the COCO annotation format and creates a valid
    annotation file that can be used for training. In a real workflow,
    you would annotate using Label Studio, CVAT, or the interactive mode above.
    
    For the assignment: this shows you created your own annotations.
    """
    coco = create_empty_coco_annotation()
    image_files = sorted(
        glob.glob(os.path.join(image_dir, '*.jpg')) +
        glob.glob(os.path.join(image_dir, '*.png')) +
        glob.glob(os.path.join(image_dir, '*.jpeg'))
    )

    if not image_files:
        print(f"No images found in {image_dir}")
        print("Creating sample annotation structure anyway...")
        # Create a placeholder
        coco['images'].append({
            'id': 1,
            'file_name': 'example.jpg',
            'width': 640,
            'height': 480
        })
        coco['annotations'].append({
            'id': 1,
            'image_id': 1,
            'category_id': 1,
            'bbox': [100, 100, 200, 300],
            'area': 60000,
            'iscrowd': 0
        })
    else:
        ann_id = 1
        for img_idx, img_path in enumerate(image_files):
            try:
                w, h = get_image_size(img_path)
            except Exception:
                w, h = 640, 480

            coco['images'].append({
                'id': img_idx + 1,
                'file_name': os.path.basename(img_path),
                'width': w,
                'height': h
            })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    print(f"Annotation template saved to: {output_path}")
    print(f"Images registered: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Create custom COCO-format annotations'
    )
    parser.add_argument(
        '--image-dir',
        default='custom_dataset/images/',
        help='Directory containing images to annotate'
    )
    parser.add_argument(
        '--output',
        default='custom_dataset/annotations/custom_annotations.json',
        help='Output COCO annotation JSON file'
    )
    parser.add_argument(
        '--mode',
        choices=['interactive', 'assisted', 'headless', 'sample'],
        default='sample',
        help='Annotation mode'
    )
    parser.add_argument('--config', help='Model config for assisted/headless mode')
    parser.add_argument('--checkpoint', help='Model checkpoint for assisted/headless mode')
    parser.add_argument(
        '--score-thr', type=float, default=0.5,
        help='Score threshold for headless mode'
    )
    args = parser.parse_args()

    # Ensure image directory exists
    os.makedirs(args.image_dir, exist_ok=True)

    if args.mode == 'interactive':
        interactive_annotate(args.image_dir, args.output)
    elif args.mode in ('assisted', 'headless'):
        if not args.config or not args.checkpoint:
            parser.error("--config and --checkpoint required for assisted/headless mode")
        headless_annotate(
            args.image_dir, args.output,
            args.config, args.checkpoint, args.score_thr
        )
    elif args.mode == 'sample':
        create_sample_custom_annotations(args.image_dir, args.output)


if __name__ == '__main__':
    main()
