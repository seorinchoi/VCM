_base_ = [
    'configs/_base_/models/segmenter_swin.py',
    'configs/_base_/datasets/KiTS21Datasets.py',
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule.py'
]

crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'

#checkpoint = '/content/drive/MyDrive/work_dirs/batch32lr0.01/epoch_120.pth'

model = dict(
    data_preprocessor=data_preprocessor,
    # backbone Swin-tiny-patch4-window7으로 수정
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        in_channels=768,
        channels = 768,# 수정된 부분
        embed_dims= 768,
        num_heads=12,
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


optimizer = dict(lr=0.001, weight_decay=0.0) #learning-rate
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)

val_evaluator = dict(type='CustomDiceMetric', target_class_index=1,iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(
    format_only= True,
    keep_results=True,
    output_dir='/content/drive/MyDrive/work_dirs/batch32lr0.001/format_results',
    iou_metrics=['mIoU'],
    type='IoUMetric')



log_level = 'DEBUG'
log_processor = dict(by_epoch=True)

