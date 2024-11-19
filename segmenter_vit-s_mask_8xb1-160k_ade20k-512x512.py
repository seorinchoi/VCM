_base_ = [
    'configs/_base_/models/segmenter_vit-b16_mask.py',
    'configs/_base_/datasets/SMC.py',
    'configs/_base_/default_runtime.py',
    'configs/_base_/schedules/schedule.py'
]
crop_size = (291, 80)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_small_p16_384_20220308-410f6037.pth'  # noqa

backbone_norm_cfg = dict(type='LN', eps=1e-6, requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(
        img_size=(291,80),
        embed_dims=384,
        num_heads=6,
        patch_size=16,
    ),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        num_classes=3,
        num_heads=3,
        num_layers=2,
        out_channels=3,
        threshold=0.3,
        channels=384,
        embed_dims=384,
        dropout_ratio=0.0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

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
