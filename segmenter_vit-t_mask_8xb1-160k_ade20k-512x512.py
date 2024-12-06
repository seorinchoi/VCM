_base_ = [
    'configs/_base_/models/segmenter_vit-b16_mask.py',
    'configs/_base_/datasets/SMC.py',
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule.py'
]
crop_size = (291, 80)
data_preprocessor = dict(size=crop_size)
checkpoint = '/content/drive/MyDrive/work_dirs/vit-t/segmenter_vit-t_mask_8x1_512x512_160k_ade20k_20220105_151706-ffcf7509.pth'  # noqa

model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=192,
        final_norm=True,
        img_size=(
            291,
            80,
        ),
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
        size=(
            291,
            80,
        ),
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
        dicttype=('FocalLoss', loss_name='loss_focal', use_sigmoid=True, loss_weight=1.0, gamma=1.5, alpha=[0.25, 0.75]),
        dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)
    ],
        norm_cfg=dict(requires_grad=True, type='LN'),
        num_classes=3,
        num_heads=3,
        num_layers=2,
        out_channels=3,
        threshold=0.3,
        type='SegmenterMaskTransformerHead'),
    #pretrained=checkpoint,

    test_cfg=dict(crop_size=(
        291,
        80,
    ), mode='slide', stride=(
        145,
        40,
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

optimizer = dict(lr=0.01, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
val_dataloader = dict(batch_size=1)
log_processor = dict(by_epoch=True)
