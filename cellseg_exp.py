# This module contains functions for running experiments including training and testing the segmentation model.

import os
import copy
from types import SimpleNamespace
import json
from tqdm import tqdm
import gc
from pathlib import Path
import importlib

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils
from torchinfo import summary
from clearml import Task

from cellseg_train_utils import (
    CombinedLoss,
    TrainEpochSchedulerStep
)
from cellseg_dataset import (
    unsplit_image,
    prepare_data_from_params
)

import matplotlib
from PIL import Image
from pprint import pprint

# Set matplotlib backend based on display availability
if os.environ.get('DISPLAY', '') == '':
    print("No display found. Using non-interactive Agg backend.")
    matplotlib.use('Agg')  # Using Agg for headless mode
else:
    matplotlib.use('TkAgg')  # Using TkAgg for normal mode

import matplotlib.pyplot as plt


def experiment(run_clear_ml=False, p=None, d=None, log_dir=None, draw=True):
    # Prepare DataLoaders if max_epochs is specified
    if p.max_epochs != -1:
        train_dataset = d.dataset_fn(d.fp_data_list.train, d.aug_list.train)
        train_loader = DataLoader(train_dataset, batch_size=p.batch_size,
                                  shuffle=True, num_workers=p.num_workers, drop_last=True)

        valid_dataset = d.dataset_fn(d.fp_data_list.valid, d.aug_list.valid)
        valid_loader = DataLoader(valid_dataset, batch_size=p.batch_size,
                                  shuffle=False, num_workers=p.num_workers, drop_last=True)
    else:
        train_loader = None
        valid_loader = None

    print(train_dataset[0][0].shape)
    print(train_dataset[0][1].shape)
    # exit()
    if log_dir is None:
        log_dir = Path('Run')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Function to freeze layers by name within the model
    def freeze_layers_by_name(model, layer_names):
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False

    # Create segmentation model with a pretrained encoder
    model_fn = getattr(importlib.import_module(
        'segmentation_models_pytorch'), p.model_name)
    model = model_fn(
        encoder_name=p.ENCODER,
        encoder_weights=p.ENCODER_WEIGHTS,
        in_channels=p.channels_num,
        classes=p.classes_num,
        activation=p.ACTIVATION,
    )
    # Alternative model creation code is commented out below
    # model = create_model_with_separate_decoder_for_boundary(model_name=p.model_name,
    #                                                         encoder_name=p.ENCODER,
    #                                                         encoder_weights=p.ENCODER_WEIGHTS,
    #                                                         in_channels=p.channels_num,
    #                                                         classes=p.classes_num - 1,
    #                                                         boundary_classes=1,
    #                                                         activation=p.ACTIVATION,
    #                                                         )

    if p.model_load_fp is None:
        print('created new')
    else:
        # Load model weights from file
        # model = torch.load(p.model_load_fp)
        pretrained_dict = torch.load(p.model_load_fp).state_dict()

        model_dict = model.state_dict()

        keys_to_load = [k for k in pretrained_dict.keys() if
                        k in model_dict.keys() and not k.startswith('segmentation_head') and not k.startswith(
                            'encoder.conv_stem')]
        # 1. Filter out unnecessary keys
        if not p.model_load_full:
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in keys_to_load}
        # 2. Update the current model state dictionary
        model_dict.update(pretrained_dict)
        # 3. Load the updated state dictionary into the model
        model.load_state_dict(model_dict)

        # Optionally freeze specified layers (commented out)
        # layers_to_freeze = [k for k in pretrained_dict.keys() if k in model_dict.keys() and k.startswith('encoder')]
        # freeze_layers_by_name(model, layers_to_freeze)
        # # Verify which layers are frozen
        # for name, param in model.named_parameters():
        #     print(name, param.requires_grad)

        print(f'loaded {p.model_load_fp}')

    summary(model)

    # Define loss and metric functions
    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    # loss = smp.losses.DiceLoss('multilabel')
    # loss.__name__ = 'dice'

    # loss = BCEDiceLoss(bce_weight=p.bce_weight)
    # loss = FocalLossMultiClass(alpha=p.focal_alpha, gamma=p.focal_gamma)
    loss = CombinedLoss(bce_weight=p.bce_weight,
                        alpha=p.focal_alpha, gamma=p.focal_gamma)
    # loss = WeightedBCEDiceLoss(bce_weight=p.bce_weight, boundary_weight=0.999)

    iou = smp.utils.metrics.IoU(threshold=0.5)
    iou.__name__ = 'IoU'

    metrics = [iou]
    for c in range(p.classes_num):
        ignore_channels = [idx for idx in range(p.classes_num) if idx != c]
        ignore_channels = torch.tensor(
            ignore_channels, dtype=torch.int64).to(p.DEVICE)
        c_iou = smp.utils.metrics.IoU(
            threshold=0.5, ignore_channels=ignore_channels)
        c_iou.__name__ = f'IoU_{c}'
        metrics.append(c_iou)

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=p.lr_first),
    ])

    # Configure learning rate scheduler: step every batch if specified
    if p.scheduler_step_every_batch:
        gamma = pow(p.lr_last / p.lr_first, 1 / (p.max_epochs *
                    len(train_loader))) if p.max_epochs != 0 else 0
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma, last_epoch=-1)
    else:
        gamma = pow(p.lr_last / p.lr_first, 1 /
                    p.max_epochs) if p.max_epochs != 0 else 0
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=gamma, last_epoch=-1)

    # Override the log formatting method for training epoch logging
    def _format_logs(self, logs):
        str_logs = ["{} - {:.4f}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    smp.utils.train.Epoch._format_logs = _format_logs

    # Create training and validation epoch runners
    train_epoch = TrainEpochSchedulerStep(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_step_every_batch=p.scheduler_step_every_batch,
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

    # Save training parameters to JSON file
    with open(log_dir / 'params.json', 'w') as f:
        p_dump = copy.copy(p)
        p_dump.dataset_dir = str(p_dump.dataset_dir)
        json.dump(vars(p_dump), f, indent=4)

    if run_clear_ml:
        task = Task.init(project_name="CellSeg4",
                         task_name=str(log_dir),
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

        valid_logs = valid_epoch.run(valid_loader)
        for k, v in valid_logs.items():
            writer.add_scalar(f'{k}/val', v, epoch)

        # Save the model if a new maximum IoU score is achieved
        if max_score < valid_logs['IoU']:
            max_score = valid_logs['IoU']
            torch.save(model, log_dir / 'best_model.pth')
            print('Model saved!')

        if not p.scheduler_step_every_batch:
            scheduler.step()

    # Testing: load best model and evaluate on test set
    print('-' * 80)
    print('TEST:')

    del train_loader
    del train_dataset
    gc.collect()

    # For testing we use the validation dataset
    test_dataset = valid_dataset
    test_loader = valid_loader

    test_epoch = smp.utils.train.ValidEpoch(
        model=model,
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

    out_dir = Path(log_dir) / 'results'

    if draw:
        iou_metric_list = draw_and_save_results(
            out_dir, test_dataset, test_loader, model, p)
        print(f'IoU = {np.mean(iou_metric_list)}')

    print('DONE!')


def test_exp(model_dir, out_dir, dataset_dir, classes, draw=True, use_all=False, batch_size=None,
             exp_class_dict=None, channels=None, ext='.jpg',
             square_a=None, border=None):
    # Test experiment: load parameters, data and model for testing the segmentation model
    model_dir = Path(model_dir)
    out_dir = Path(out_dir)

    with open(model_dir / 'params.json', 'r') as f:
        cfg = json.load(f)
        p = SimpleNamespace(**cfg)
        p.dataset_dir = Path(dataset_dir)

    if exp_class_dict is not None:
        p.exp_class_dict = exp_class_dict

    if channels is not None:
        p.channels = channels
        
    p.model_load_fp = model_dir / 'best_model.pth'

    if use_all:
        p.ratio_train = 0.0
        p.ratio_val = 0.0
        p.images_num = None

    pprint(vars(p))

    if square_a is not None and border is not None:
        p.square_a, p.border = square_a, border

    fp_data_list, aug_list, dataset_fn, dataset_test_fn = prepare_data_from_params(p, classes,
                                                                                   shuffle=True,
                                                                                   max_workers=1,
                                                                                   ext=ext, resize_coef=1)

    d = SimpleNamespace(fp_data_list=fp_data_list,
                        aug_list=aug_list,
                        dataset_fn=dataset_test_fn)

    if batch_size:
        p.batch_size = batch_size
    test_dataset = d.dataset_fn(d.fp_data_list.test, d.aug_list.test)
    test_loader = DataLoader(test_dataset, batch_size=p.batch_size,
                             shuffle=False, num_workers=1, drop_last=True)

    # Function to freeze layers by name (if needed)
    def freeze_layers_by_name(model, layer_names):
        for name, param in model.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False

    # Create segmentation model with the specified configuration
    model_fn = getattr(importlib.import_module(
        'segmentation_models_pytorch'), p.model_name)
    model = model_fn(
        encoder_name=p.ENCODER,
        encoder_weights=p.ENCODER_WEIGHTS,
        in_channels=p.channels_num,
        classes=p.classes_num,
        activation=p.ACTIVATION,
    )

    # model = torch.load(p.model_load_fp)
    pretrained_dict = torch.load(str(p.model_load_fp)).state_dict()

    # pretrained_dict = {k: v.cpu() for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()

    keys_to_load = [k for k in pretrained_dict.keys() if
                    k in model_dict.keys() and not k.startswith('segmentation_head') and not k.startswith(
                        'encoder.conv_stem')]
    if not p.model_load_full:
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in keys_to_load}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    print(f'loaded {p.model_load_fp}')

    summary(model)

    loss = CombinedLoss(bce_weight=p.bce_weight,
                        alpha=p.focal_alpha, gamma=p.focal_gamma)
    iou = smp.utils.metrics.IoU(threshold=0.5)
    iou.__name__ = 'IoU'

    metrics = [iou]
    for c in range(p.classes_num):
        ignore_channels = [idx for idx in range(p.classes_num) if idx != c]
        ignore_channels = torch.tensor(
            ignore_channels, dtype=torch.int64).to(p.DEVICE)
        c_iou = smp.utils.metrics.IoU(
            threshold=0.5, ignore_channels=ignore_channels)
        c_iou.__name__ = f'IoU_{c}'
        metrics.append(c_iou)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4f}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    # smp.utils.train.Epoch._format_logs = _format_logs
    # test_epoch = smp.utils.train.ValidEpoch(
    #     model=model,
    #     loss=loss,
    #     metrics=metrics,
    #     device=p.DEVICE,
    # )

    # test_logs = test_epoch.run(test_loader)
    # for k, v in test_logs.items():
    #     print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

    model = model.to(p.DEVICE)

    # torch.save(model, str(p.model_load_fp))
    # exit()

    if draw:
        iou_metric_list = draw_and_save_results(
            out_dir, test_dataset, test_loader, model, p)
        print(f'IoU = {np.mean(iou_metric_list)}')

    print('DONE!')


def draw_and_save_results(out_dir, test_dataset, test_loader, model, p):
    # Visualize and save the results of the best model on the test dataset
    iou_metric_list = list()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'full').mkdir(parents=True, exist_ok=True)
    (out_dir / 'compare').mkdir(parents=True, exist_ok=True)
    (out_dir / 'predicted_masks').mkdir(parents=True, exist_ok=True)

    w = h = 20
    img_num = p.channels_num + 2
    figsize = (w * img_num, h)

    img_list = list()
    gt_list = list()
    pr_list = list()
    info_list = list()

    class_num = 1 if test_dataset.classes is None else len(
        test_dataset.classes)
    class_num += 1
    squares = test_dataset.squares
    border = test_dataset.border

    plt.ioff()
    pbar = tqdm(total=int(len(test_loader.dataset) / len(squares)))
    for batch_idx, (x, y) in enumerate(test_loader):
        pred_y = model.predict(x.to(p.DEVICE))
        test_loader.dataset.__getitem__(batch_idx * p.batch_size)
        for img_idx in range(y.shape[0]):
            n = batch_idx * p.batch_size + img_idx
            info = test_dataset.info[n]

            image = x[img_idx].squeeze().cpu().numpy()
            gt_mask = y[img_idx].squeeze().cpu().numpy().round()
            pr_mask = pred_y[img_idx].squeeze().cpu().numpy().round()

            if len(image.shape) == 2:
                image = image[np.newaxis, ...]
            if len(gt_mask.shape) == 2:
                gt_mask = gt_mask[np.newaxis, ...]
            if len(pr_mask.shape) == 2:
                pr_mask = pr_mask[np.newaxis, ...]

            img_list.append(image)
            gt_list.append(gt_mask)
            pr_list.append(pr_mask)
            info_list.append(info)

            iou_metric = smp.utils.functional.iou(
                torch.from_numpy(gt_mask), torch.from_numpy(pr_mask))
            iou_metric_list.append(iou_metric)

        # Process accumulated images once enough samples are collected
        if (len(img_list) >= len(squares)) or (batch_idx == len(test_loader) - 1):
            res_idx = info_list[0]['idx']
            exp_dir = info_list[0]['exp_dir']
            (out_dir / 'full' / exp_dir).mkdir(exist_ok=True)
            (out_dir / 'compare' / exp_dir).mkdir(exist_ok=True)
            (out_dir / 'predicted_masks' / exp_dir).mkdir(exist_ok=True)

            print(f'save results for: {res_idx}')

            assert all(
                [info['idx'] == res_idx for info in info_list[:len(squares)]])

            restored_img = unsplit_image(img_list[:len(squares)],
                                         squares,
                                         'square_coords',
                                         border)
            restored_gt = unsplit_image(gt_list[:len(squares)],
                                        squares,
                                        'square_coords',
                                        border)
            restored_pr = unsplit_image(pr_list[:len(squares)],
                                        squares,
                                        'square_coords',
                                        border)

            for idx in range(restored_pr.shape[0]):
                res_idx_name = f'{res_idx}_{idx}'
                img = Image.fromarray(restored_pr[idx].astype(np.uint8) * 255)
                img = img.resize(test_loader.dataset.target_size)
                img.save((out_dir / 'predicted_masks' /
                         exp_dir / res_idx_name).with_suffix('.png'))

            restored_gt_full = restored_gt[0]
            for idx in range(restored_gt.shape[0]):
                restored_gt_full[restored_gt[idx] == 1] = idx + 1
            restored_pr_full = restored_pr[0]
            for idx in range(restored_pr.shape[0]):
                restored_pr_full[restored_pr[idx] == 1] = idx + 1

            fig, ax = plt.subplots(1, img_num, figsize=figsize)
            for c_idx in range(p.channels_num):
                ax[c_idx].imshow(restored_img[c_idx],
                                 cmap='gray', vmin=0, vmax=1)
                ax[c_idx].title.set_text(f'# {c_idx}')
            ax[p.channels_num +
                0].imshow(restored_gt_full, cmap='gray', vmin=0, vmax=class_num)
            ax[p.channels_num + 0].title.set_text('gt_mask')
            ax[p.channels_num +
                1].imshow(restored_pr_full, cmap='gray', vmin=0, vmax=class_num)
            ax[p.channels_num + 1].title.set_text('pr_mask')

            fig.savefig(
                (out_dir / 'full' / exp_dir/ res_idx).with_suffix('.png'), bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 2, figsize=(w * 2, h))
            color_shift_red = (+100, -100, -100)
            color_shift_green = (-100, +100, -100)
            color_shift_blue = (-100, -100, +100)
            # color_shift_violet = (+100, -100, +100)
            color_shift_yellow = (+100, +100, -100)

            img_gt = restored_img[0].copy()
            img_gt3 = np.stack([img_gt, img_gt, img_gt], axis=2) * 255
            red_idx = restored_gt[0] == 1
            for c_idx, c in enumerate(color_shift_red):
                img_gt3[..., c_idx][red_idx] += c
            if restored_gt.shape[0] > 2:
                blue_idx = restored_gt[1] == 1
                for c_idx, c in enumerate(color_shift_blue):
                    img_gt3[..., c_idx][blue_idx] += c
                yellow_idx = restored_gt[2] == 1
                for c_idx, c in enumerate(color_shift_yellow):
                    img_gt3[..., c_idx][yellow_idx] += c
            green_idx = restored_gt[-1] == 1
            for c_idx, c in enumerate(color_shift_green):
                img_gt3[..., c_idx][green_idx] += c

            np.clip(img_gt3, 0, 255, out=img_gt3)
            img_gt3 = img_gt3.astype(np.uint8)
            ax[0].imshow(img_gt3)
            ax[0].title.set_text('img_gt')

            img_pr = restored_img[0].copy()
            img_pr3 = np.stack([img_pr, img_pr, img_pr], axis=2) * 255
            red_idx = restored_pr[0] == 1
            for c_idx, c in enumerate(color_shift_red):
                img_pr3[..., c_idx][red_idx] += c

            if restored_pr.shape[0] > 2:
                blue_idx = restored_pr[1] == 1
                for c_idx, c in enumerate(color_shift_blue):
                    img_pr3[..., c_idx][blue_idx] += c
                yellow_idx = restored_pr[2] == 1
                for c_idx, c in enumerate(color_shift_yellow):
                    img_pr3[..., c_idx][yellow_idx] += c
            green_idx = restored_pr[-1] == 1
            for c_idx, c in enumerate(color_shift_green):
                img_pr3[..., c_idx][green_idx] += c

            np.clip(img_pr3, 0, 255, out=img_pr3)
            img_pr3 = img_pr3.astype(np.uint8)
            ax[1].imshow(img_pr3)
            ax[1].title.set_text('img_pr')

            fig.savefig(
                (out_dir / 'compare' / exp_dir / res_idx).with_suffix('.png'), bbox_inches='tight')
            plt.close(fig)

            img_list = img_list[len(squares):]
            gt_list = gt_list[len(squares):]
            pr_list = pr_list[len(squares):]
            info_list = info_list[len(squares):]
            pbar.update(1)

            # Manually trigger garbage collection to free memory
            gc.collect()
    pbar.close()
    return iou_metric_list
