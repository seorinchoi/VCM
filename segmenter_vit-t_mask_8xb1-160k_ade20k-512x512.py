_base_ = [
    'configs/_base_/models/segmenter_vit-b16_mask.py',
    'configs/_base_/datasets/SMC.py',
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule.py'
]
crop_size = (80, 291)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa

model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=192,
        final_norm=True,
        img_size=crop_size,
        in_channels=3,
        interpolate_mode='bicubic',
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_heads=3,
        num_layers=12,
        patch_size=16, #patch size 16
        type='VisionTransformer',
        with_cls_token=True),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            127.5,
            127.5,
            127.5,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=crop_size,
        std=[
            127.5,
            127.5,
            127.5,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        channels=192,
        dropout_ratio=0.0,
        embed_dims=192,
        in_channels=192,
        loss_decode=[
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
        dict(type='DiceLoss', loss_name='loss_dice', use_sigmoid=False, loss_weight=3.0)
    ],
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_classes=2,
        num_heads=3,
        num_layers=2,
        out_channels=2,
        threshold=0.3,
        type='SegmenterMaskTransformerHead'),
    #pretrained=checkpoint,

    test_cfg=dict(crop_size=(
        80,
        291,
    ), mode='slide', stride=(
        80,
        291,
    )),
    type='EncoderDecoder')

train_dataloader = dict(batch_size=16) #batch-size
val_dataloader = dict(batch_size=1)


val_evaluator = dict(type='CustomDiceMetric', target_class_index=1,iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(
    format_only= True,
    keep_results=True,
    output_dir='',
    iou_metrics=['mIoU'],
    type='IoUMetric')


#load_from=checkpoint
#resume_from=

optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
val_dataloader = dict(batch_size=1)
log_processor = dict(by_epoch=True)
