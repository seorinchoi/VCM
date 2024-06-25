# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=80,  # 80 epochs
        by_epoch=True)
]

# training schedule for 80 epochs
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=430, val_interval=1)  # validate every 10 epochs
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# default hooks including early stopping
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # IterTimerHook을 유지
    logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),  # save checkpoint every 10 epochs
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='loss',  # Metric to monitor
        patience=10,  # Number of epochs to wait for improvement
        min_delta=0.01  # Minimum change to qualify as an improvement
    )
)
