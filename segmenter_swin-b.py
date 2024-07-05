_base_ = [
    'configs/_base_/models/segmenter_swin.py',
    'configs/_base_/datasets/ade20k.py',
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

checkpoint = 'hhttps://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    # backbone Swin-base-patch4-window7으로 수정
    backbone=dict(
        init_cfg=dict(
            checkpoint= checkpoint,
            type='Pretrained',
            strict=False),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=1024,
        channels = 1024,# swin-b
        embed_dims= 1024,
        num_heads=16,
        num_layers=2,
        out_channels=3,
        dropout_ratio=0.0,
        num_classes=3,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=(256,256), stride=(128, 128)),
)


train_dataloader = dict(batch_size=32) #batch-size
val_dataloader = dict(batch_size=1)


optimizer = dict(lr=0.01, weight_decay=0.0) #learning-rate
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

val_evaluator = dict(type='CustomDiceMetric', target_class_index=1,iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(
    format_only= True,
    keep_results=True,
    output_dir='work_dirs/Swin-Seg/batch32lr0.01/format_results',
    iou_metrics=['mIoU','mDice'],
    target_class_index=1,
    type='CustomDiceMetric')

log_level = 'DEBUG'
log_processor = dict(by_epoch=True)
