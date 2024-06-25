# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook, EarlyStoppingHook)
from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from mmengine.runner.loops import EpochBasedTrainLoop, TestLoop, ValLoop
from torch.optim.sgd import SGD

from mmseg.engine import SegVisualizationHook


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
    type='EpochBasedTrainLoop', max_epochs=300, val_interval=10)  # validate every 10 epochs
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# default hooks including early stopping
default_hooks = dict(
    timer=dict(type='IterTimerHook'),  # IterTimerHook을 유지
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=10),  # save checkpoint every 10 epochs
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='mIoU',  # Metric to monitor
        patience=10,  # Number of epochs to wait for improvement
        interval=1,  # Check interval
        min_delta=0.01  # Minimum change to qualify as an improvement
    )
)
