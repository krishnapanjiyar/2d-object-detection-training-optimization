# Custom Dataset Annotation Guide

## Overview

This guide explains how to create your own annotated dataset for the training optimization project.
You need to provide **your own annotations** — simply downloading a pre-annotated dataset does not count.

Three methods are provided (choose one):

---

## Method 1: Label Studio (Recommended for Submission)

Label Studio is a professional annotation tool that provides a web UI for drawing bounding boxes.

### Setup

```bash
# Install Label Studio
pip install label-studio

# Start the server
label-studio start --port 8080
```

Then open http://localhost:8080 in your browser.

### Steps

1. **Create a project** → Select "Object Detection with Bounding Boxes"

2. **Import your images**:
   - Click "Import" and upload images from `custom_dataset/images/`
   - Or connect a local directory

3. **Configure labels** — Use the labeling interface XML:
   ```xml
   <View>
     <Image name="image" value="$image"/>
     <RectangleLabels name="label" toName="image">
       <Label value="person" background="red"/>
       <Label value="car" background="blue"/>
       <Label value="bicycle" background="green"/>
       <Label value="dog" background="yellow"/>
       <Label value="cat" background="purple"/>
       <!-- Add more categories as needed -->
     </RectangleLabels>
   </View>
   ```

4. **Annotate**: Draw bounding boxes around objects in each image

5. **Export**:
   - Go to project settings → Export
   - Select "COCO" format
   - Download and save as `custom_dataset/annotations/custom_annotations.json`

### Screenshot for Submission
Take a screenshot of the Label Studio interface showing your annotations.
This proves you created annotations yourself.

---

## Method 2: Interactive Script (CLI)

Use the provided annotation script with OpenCV:

```bash
# Install opencv
pip install opencv-python

# Run interactive annotator
python scripts/create_custom_annotations.py \
    --image-dir custom_dataset/images/ \
    --output custom_dataset/annotations/custom_annotations.json \
    --mode interactive
```

Controls:
- **Click + drag**: Draw bounding box
- **Enter**: Confirm and pick category
- **n**: Next image
- **u**: Undo last box
- **q**: Save and quit

---

## Method 3: Model-Assisted + Manual Correction

1. Run a pretrained model on your images to generate initial predictions:

```bash
python scripts/create_custom_annotations.py \
    --image-dir custom_dataset/images/ \
    --output custom_dataset/annotations/custom_annotations_draft.json \
    --mode headless \
    --config configs/baseline_faster_rcnn_r50_fpn.py \
    --checkpoint work_dirs/baseline_r50_fpn_1x/epoch_12.pth \
    --score-thr 0.5
```

2. Import the draft annotations into Label Studio for correction:
   - Create a Label Studio project
   - Import the JSON and images
   - Review each image: fix wrong labels, adjust boxes, delete false positives, add missed objects
   - Export corrected annotations

---

## Collecting Your Own Images

You need **at least 20-50 images** with objects you want to detect.

Options for collecting images:
- **Take photos** with your phone (best for proving originality)
- **Record a short video** and extract frames:
  ```bash
  # Extract 1 frame per second from a video
  ffmpeg -i video.mp4 -vf "fps=1" custom_dataset/images/frame_%04d.jpg
  ```
- **Use a webcam**:
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  for i in range(50):
      ret, frame = cap.read()
      cv2.imwrite(f'custom_dataset/images/img_{i:04d}.jpg', frame)
      cv2.waitKey(500)  # 0.5s between captures
  cap.release()
  ```

---

## After Annotation

### Train with custom data:
```bash
python tools/train.py configs/finetune_with_custom_data.py \
    --work-dir work_dirs/finetuned_custom
```

### Verify annotations are valid:
```python
import json
with open('custom_dataset/annotations/custom_annotations.json') as f:
    coco = json.load(f)
print(f"Images: {len(coco['images'])}")
print(f"Annotations: {len(coco['annotations'])}")
print(f"Categories: {len(coco['categories'])}")
```

---

## Expected Directory Structure

```
custom_dataset/
├── images/
│   ├── img_0001.jpg
│   ├── img_0002.jpg
│   └── ...
└── annotations/
    └── custom_annotations.json     ← COCO format
```
