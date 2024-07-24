# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
# load_from = 'https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'  # noqa

pretrained = '/home/chli/Model/Swin/swin_large_patch4_window12_384_22k.pth'
# load_from = '/home/chli/Model/Co-DETR/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
load_from = '/home/chli/github/XRay/mm-detection/output/co_detr/xray-v1.pth'
resume = False

data_root = '/home/chli/Dataset/X-Ray/'
train_json = 'train/annotations/instances_default.json'
val_json = 'val/annotations/instances_default.json'
dataset_type = 'CocoDataset'
num_classes = 8
image_size = (640, 640)
metainfo = {
    'classes': ('knife', 'wrench', 'glassbottle', 'drinkbottle', 'powerbank', 'umbrella', 'metalcup', 'lighter'),
    'palette': [
        (220, 20, 60),
        (20, 220, 60),
        (20, 20, 250),
        (120, 120, 60),
        (120, 20, 160),
        (20, 120, 160),
        (20, 100, 0),
        (0, 20, 100),
    ]
}

num_dec_layer = 6
loss_lambda = 2.0
max_epochs = 32
max_iters = 27000000
batch_size = 3
num_workers = 3
num_gpu = 1
lr = 1e-5

work_dir = '/home/chli/github/XRay/mm-detection/output/co_detr'

default_scope = 'mmdet'
backend_args = None
log_level = 'INFO'
logger_interval = 50
max_keep_ckpts = 3

custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.CO-DETR.codetr',
    ])

batch_augments = [
    dict(
        pad_mask=False,
        size=image_size,
        type='BatchFixedSizePad'
    ),
]

default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=max_keep_ckpts,
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=logger_interval, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'

model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[2, 2, 18, 2,],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(
            checkpoint=pretrained,
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[6, 12, 24, 48,],
        out_indices=(0, 1, 2, 3,),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=12,
        with_cp=True),
    bbox_head=[
        dict(
            anchor_generator=dict(
                octave_base_scale=8,
                ratios=[
                    1.0,
                ],
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128,],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[0.0, 0.0, 0.0, 0.0,],
                target_stds=[0.1, 0.1, 0.2, 0.2,],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=2.0 * num_dec_layer * loss_lambda, type='GIoULoss'),
            loss_centerness=dict(
                loss_weight=1.0 * num_dec_layer * loss_lambda, type='CrossEntropyLoss', use_sigmoid=True),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=num_classes,
            stacked_convs=1,
            type='CoATSSHead'),
    ],
    data_preprocessor=dict(
        batch_augments=batch_augments,
        bgr_to_rgb=True,
        mean=[123.675, 116.28, 103.53,],
        pad_mask=False,
        std=[58.395, 57.12, 57.375,],
        type='DetDataPreprocessor'),
    eval_module='detr',
    neck=dict(
        act_cfg=None,
        in_channels=[192, 384, 768, 1536,],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
    query_head=dict(
        as_two_stage=True,
        dn_cfg=dict(
            box_noise_scale=0.4,
            group_cfg=dict(dynamic=True, num_dn_queries=500, num_groups=None),
            label_noise_scale=0.5),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=num_classes,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            temperature=20,
            type='SinePositionalEncoding'),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_levels=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='DinoTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        dropout=0.0,
                        embed_dims=256,
                        num_levels=5,
                        type='MultiScaleDeformableAttention'),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder',
                with_cp=6),
            num_co_heads=2,
            num_feature_levels=5,
            type='CoDinoTransformer',
            with_coord_feat=True),
        type='CoDINOHead'),
    roi_head=[
        dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[0.0, 0.0, 0.0, 0.0,],
                    target_stds=[0.1, 0.1, 0.2, 0.2,],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=10.0 * num_dec_layer * loss_lambda, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=1.0 * num_dec_layer * loss_lambda,
                    type='CrossEntropyLoss',
                    use_sigmoid=True),
                num_classes=num_classes,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[4, 8, 16, 32, 64,],
                finest_scale=56,
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='CoStandardRoIHead'),
    ],
    rpn_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[0.5, 1.0, 2.0,],
            scales_per_octave=3,
            strides=[4, 8, 16, 32, 64, 128,],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[0.0, 0.0, 0.0, 0.0,],
            target_stds=[1.0, 1.0, 1.0, 1.0,],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0 * num_dec_layer * loss_lambda, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0 * num_dec_layer * loss_lambda, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=[
        dict(max_per_img=300, nms=dict(iou_threshold=0.8, type='soft_nms')),
        dict(
            rcnn=dict(
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.0),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        dict(
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.0),
    ],
    train_cfg=[
        dict(
            assigner=dict(
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='HungarianAssigner')),
        dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=4000)),
        dict(
            allowed_border=-1,
            assigner=dict(topk=9, type='ATSSAssigner'),
            debug=False,
            pos_weight=-1),
    ],
    type='CoDETR',
    use_lsj=True,
)

load_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        keep_ratio=True,
        ratio_range=(0.8, 1.2,),
        scale=image_size,
        type='RandomResize'),
    dict(
        allow_negative_crop=True,
        crop_size=image_size,
        crop_type='absolute_range',
        recompute_bbox=True,
        type='RandomCrop'),
    dict(min_gt_bbox_wh=(1e-2, 1e-2,),
         type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(pad_val=dict(img=(114, 114, 114,)), size=image_size,
    type='Pad'),
]

train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (600, 600,), (600, 500,), (500, 600,), (600, 400,),
                        (400, 600,), (500, 400,), (400, 500,), (600, 300,),
                        (300, 600,),
                    ],
                    type='RandomChoiceResize'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (400, 600,), (500, 600,), (600, 600,),
                    ],
                    type='RandomChoiceResize'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(384, 600,),
                    crop_type='absolute_range',
                    type='RandomCrop'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (600, 600,), (600, 500,), (500, 600,), (600, 400,),
                        (400, 600,), (500, 400,), (400, 500,), (600, 300,),
                        (300, 600,),
                    ],
                    type='RandomChoiceResize'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        ann_file=train_json,
        backend_args=None,
        data_prefix=dict(img='train/images/'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        metainfo=metainfo,
        pipeline=train_pipeline,
        type=dataset_type),
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))

val_cfg = dict(_scope_='mmdet', type='ValLoop')

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=image_size, type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        _scope_='mmdet',
        ann_file=val_json,
        backend_args=None,
        data_prefix=dict(img='val/images/'),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=val_pipeline,
        test_mode=True,
        type=dataset_type),
    drop_last=False,
    num_workers=num_workers,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    _scope_='mmdet',
    ann_file= data_root + val_json,
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')

test_cfg = dict(_scope_='mmdet', type='TestLoop')

test_pipeline = val_pipeline


test_dataloader = val_dataloader

test_evaluator = val_evaluator

optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=lr, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')

auto_scale_lr = dict(base_batch_size=num_gpu*batch_size)

param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=max_epochs,
        gamma=0.1,
        milestones=[
            int(max_epochs/2),
        ],
        type='MultiStepLR'),
]

log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)

vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]

visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='Visualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
