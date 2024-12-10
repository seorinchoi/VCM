# 데이터셋 타입과 경로를 수정
dataset_type = 'SMCDatasets'
data_root = '/content/drive/MyDrive/SMC/'
test_data_root = '/content/drive/MyDrive/SMC/test/'
test_img_dir = 'images'
test_mask_dir = 'labels'
img_dir = 'train/images'
ann_dir = 'train/labels'

crop_size = (80,291)

#train, validation 파이프라인
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False ),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs'),
]

# test 파이프라인 설정
test_pipeline = val_pipeline
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
]
train_dataloader = dict(
    batch_size=8, #배치사이즈 24
    dataset=dict(
        type='SMCDatasets',
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=train_pipeline,
        reduce_zero_label=False,  # 여기서 reduce_zero_label 설정
        ann_file='splits/train.txt', 
        ignore_index=255,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='SMCDatasets',
        data_root=data_root,
        data_prefix=dict(img_path=img_dir, seg_map_path=ann_dir),
        pipeline=val_pipeline,
        reduce_zero_label=False,  # 여기서 reduce_zero_label 설정
        ann_file='splits/val.txt',
        ignore_index=255,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
test_dataloader =  dict(
    batch_size=1,
    dataset=dict(
        type='SMCDatasets',
        data_root=test_data_root,
        data_prefix=dict(img_path=test_img_dir, seg_map_path=test_mask_dir),
        pipeline=test_pipeline,
        reduce_zero_label=False,  # 여기서 reduce_zero_label 설정
        ann_file='splits/test.txt',
        ignore_index=255,
    ),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
