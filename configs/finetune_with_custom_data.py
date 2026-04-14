# Config: Fine-tune optimized model with custom annotated data
# 
# This config demonstrates how to incorporate custom annotations
# into training. It uses ConcatDataset to merge COCO with your
# custom annotated images, then fine-tunes the optimized model.
#
# Workflow:
#   1. Annotate your images (scripts/create_custom_annotations.py)
#   2. Place images in custom_dataset/images/
#   3. Run: python tools/train.py configs/finetune_with_custom_data.py
#
# The custom annotations must follow COCO format (see create_custom_annotations.py)

_base_ = ['./optimized_faster_rcnn_r101_pafpn.py']

# ========================
# DATASET: COCO + Custom Data (ConcatDataset)
# ========================
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
custom_data_root = 'custom_dataset/'

# Reuse the optimized augmentation pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=[(1333, 480), (1333, 800)],
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

# COCO training dataset
coco_train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

# Custom annotated dataset
custom_train_dataset = dict(
    type=dataset_type,
    data_root=custom_data_root,
    ann_file='annotations/custom_annotations.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

# Merge datasets using ConcatDataset
# Custom data is repeated 5x to give it more weight relative to COCO
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            coco_train_dataset,
            # Repeat custom data to increase its representation
            custom_train_dataset,
            custom_train_dataset,
            custom_train_dataset,
            custom_train_dataset,
            custom_train_dataset,
        ]
    )
)

# Fine-tuning: start from optimized checkpoint, train 6 more epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=6, val_interval=1)

# Lower learning rate for fine-tuning
optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00002,                     # 5x lower than initial training
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    clip_grad=dict(max_norm=35, norm_type=2),
    loss_scale='dynamic'
)

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=6,
        eta_min=1e-7
    )
]

# Load from the optimized model checkpoint
load_from = 'work_dirs/optimized_r101_pafpn_2x/best_coco_bbox_mAP_epoch_24.pth'
