_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0)
# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='mmcls.ResArch',
        arch='IRNet-18',
        binary_type=(True, True),
        out_indices=(0,1,2, 3),
        frozen_stages=1,
        stem_act='hardtanh',
        style='pytorch',
                ),
    neck=dict(
        type='FPN',
        in_channels=[64,128,256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

find_unused_parameters=True
work_dir = 'work_dirs/retinanet_resnet_fpn_1x_coco_2x_1'
load_from  = '/workspace/S/jiangfei/BinaryNeuralNetwork_debug/work_dirs/irnet_r18_bias_x2x_bnall/epoch_100.pth'
seed = 303
