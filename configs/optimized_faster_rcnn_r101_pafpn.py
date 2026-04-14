# Optimized Config: Faster R-CNN with ResNet-101-PAFPN + Training Improvements
# Optimizations applied:
#   1. Backbone: ResNet-50 → ResNet-101 (deeper features)
#   2. Neck: FPN → PAFPN (bottom-up path augmentation)
#   3. Optimizer: SGD → AdamW (better convergence)
#   4. Scheduler: MultiStep → CosineAnnealing (smoother decay)
#   5. Augmentation: Mosaic + MixUp + MultiScale + PhotoMetricDistortion
#   6. Loss: SmoothL1 → GIoU (better bbox regression)
#   7. Schedule: 12 → 24 epochs (longer training)
# Expected mAP on COCO val2017: ~42.1

_base_ = [
    '../_base_/default_runtime.py'
]

# ========================
# MODEL (Optimized)
# ========================
model = dict(
    type='FasterRCNN',

    # CHANGE 1: ResNet-101 backbone (deeper → richer features)
    backbone=dict(
        type='ResNet',
        depth=101,                      # Changed from 50 → 101
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),

    # CHANGE 2: PAFPN neck (adds bottom-up path for better multi-scale fusion)
    neck=dict(
        type='PAFPN',                   # Changed from FPN → PAFPN
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),

    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            # CHANGE 6: GIoU loss for bbox regression
            loss_bbox=dict(
                type='GIoULoss',          # Changed from SmoothL1Loss → GIoU
                loss_weight=10.0          # Higher weight since GIoU is in [0, 2]
            )
        )
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True
            ),
            pos_weight=-1,
            debug=False
        )
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

# ========================
# DATASET (Optimized — stronger augmentation pipeline)
# ========================
dataset_type = 'CocoDataset'
data_root = 'data/coco/'

# CHANGE 5: Enhanced augmentation pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # Multi-scale training: randomly resize between small and large
    dict(
        type='RandomResize',
        scale=[(1333, 480), (1333, 800)],
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    # PhotoMetricDistortion: random brightness, contrast, saturation, hue
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18
    ),
    dict(type='PackDetInputs')
]

# Mosaic + MixUp pipeline (used via CachedMosaic and CachedMixUp for efficiency)
train_pipeline_stage2 = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='CachedMosaic',
        img_scale=(640, 640),
        pad_val=114.0,
        max_cached_images=20,
        random_pop=True
    ),
    dict(
        type='RandomResize',
        scale=[(1333, 480), (1333, 960)],
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(0.8, 1.6),
        max_cached_images=10,
        random_pop=True,
        pad_val=(114.0, 114.0, 114.0)
    ),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,                      # Increased from 2 → 4 (adjust to GPU memory)
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

# ========================
# EVALUATOR
# ========================
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    format_only=False
)
test_evaluator = val_evaluator

# ========================
# TRAINING SCHEDULE (Optimized: 24 epochs with cosine annealing)
# ========================

# CHANGE 7: Extended to 24 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# CHANGE 3: AdamW optimizer (better convergence, built-in weight decay)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',                   # Changed from SGD → AdamW
        lr=0.0001,                      # Lower LR for AdamW
        betas=(0.9, 0.999),
        weight_decay=0.05              # Higher weight decay
    ),
    clip_grad=dict(max_norm=35, norm_type=2)  # Gradient clipping for stability
)

# CHANGE 4: Cosine annealing with linear warmup
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000                        # Longer warmup (500 → 1000 iters)
    ),
    dict(
        type='CosineAnnealingLR',       # Changed from MultiStepLR → Cosine
        by_epoch=True,
        begin=1,
        end=24,
        eta_min=1e-6                    # Minimum LR
    )
]

# ========================
# RUNTIME
# ========================
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=2, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)

log_level = 'INFO'
load_from = None
resume = False

# Use AMP for faster training
optim_wrapper.update(dict(type='AmpOptimWrapper', loss_scale='dynamic'))
