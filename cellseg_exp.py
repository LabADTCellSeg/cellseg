import os
# import os.path as osp
import importlib

# import nd2
# import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from torchinfo import summary

from pprint import pprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib

from clearml import Task

from cellseg_utils import BCEDiceLoss, unsplit_image

matplotlib.use('TkAgg')


def experiment(run_clear_ml=False, p=None, d=None, log_dir=None, draw=True):
    if p.max_epochs != 0:
        train_loader = DataLoader(d.train_dataset, batch_size=p.batch_size,
                                  shuffle=True, num_workers=p.num_workers)
        valid_loader = DataLoader(d.valid_dataset, batch_size=p.batch_size,
                                  shuffle=False, num_workers=p.num_workers)
    test_loader = DataLoader(d.test_dataset, batch_size=16,
                             shuffle=False)

    if log_dir is None:
        log_dir = 'Run'
    os.makedirs(log_dir)

    # Freeze layers by name
    def freeze_layers_by_name(model, layer_names):
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False

    # create segmentation model with pretrained encoder
    model_func = getattr(importlib.import_module('segmentation_models_pytorch'),
                         p.model_name)
    model = model_func(
        encoder_name=p.ENCODER,
        encoder_weights=p.ENCODER_WEIGHTS,
        in_channels=p.channels,
        classes=p.classes_num,
        activation=p.ACTIVATION,
    )
    if p.model_load_fp is None:
        print('created new')
    else:
        # model = torch.load(p.model_load_fp)
        pretrained_dict = torch.load(p.model_load_fp).state_dict()

        model_dict = model.state_dict()

        keys_to_load = [k for k in pretrained_dict.keys() if k in model_dict.keys() and not k.startswith('segmentation_head') and not k.startswith('encoder.conv_stem')]
        # 1. filter out unnecessary keys
        if not p.model_load_full:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in keys_to_load}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)

        # Freeze the specified layers
        layers_to_freeze = [k for k in pretrained_dict.keys() if k in model_dict.keys() and k.startswith('encoder')]
        # freeze_layers_by_name(model, layers_to_freeze)

        # # Verify which layers are frozen
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        print(f'loaded {p.model_load_fp}')

    summary(model)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    # loss = smp.losses.DiceLoss('multilabel')
    # loss.__name__ = 'dice'

    loss = BCEDiceLoss(bce_weight=p.bce_weight)

    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=p.lr_first),
    ])

    # a, b = next(iter(test_loader))
    # ans = loss(b, b)
    # print(ans)
    # create epoch runners
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=p.DEVICE,
        verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=p.DEVICE,
        verbose=True,
    )

    # if p.max_epochs != 0:
    gamma = pow(p.lr_last / p.lr_first, 1 / p.max_epochs) if p.max_epochs != 0 else 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    # TRAIN
    if run_clear_ml:
        task = Task.init(project_name="CellSeg4",
                         task_name=log_dir,
                         output_uri=False)
        task.connect(vars(p))
    else:
        task = None
    writer = SummaryWriter(log_dir=log_dir)

    max_score = 0
    for epoch in range(0, p.max_epochs):
        print(f'Epoch: #{epoch} / {p.max_epochs}')
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        train_logs = train_epoch.run(train_loader)
        for k, v in train_logs.items():
            writer.add_scalar(f'{k}/train', v, epoch)
            # print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

        valid_logs = valid_epoch.run(valid_loader)
        for k, v in valid_logs.items():
            writer.add_scalar(f'{k}/val', v, epoch)
            # print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, os.path.join(log_dir, 'best_model.pth'))
            print('Model saved!')

        scheduler.step()

    # load best saved checkpoint
    print('-' * 80)
    print('TEST:')
    best_model = model

    # evaluate model on test set
    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=p.DEVICE,
    )

    test_logs = test_epoch.run(test_loader)
    for k, v in test_logs.items():
        writer.add_scalar(f'{k}/test', v, 0)
        print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

    writer.close()
    if run_clear_ml:
        task.close()