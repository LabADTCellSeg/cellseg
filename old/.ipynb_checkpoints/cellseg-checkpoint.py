import os
import os.path as osp
import importlib

import nd2
import numpy as np

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
        task = Task.init(project_name="CellSeg",
                         task_name=log_dir,
                         output_uri=True)
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

    if draw:
        # visualize results of best saved model
        iou_metric_list = list()

        out_dir = os.path.join(log_dir, 'results')
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(osp.join(out_dir, 'full'), exist_ok=True)
        os.makedirs(osp.join(out_dir, 'compare'), exist_ok=True)

        # for directory in ['image', 'gt', 'pr', 'compare']:
        #     os.makedirs(os.path.join(out_dir, directory), exist_ok=True)

        W = H = 20
        img_num = p.channels + 2
        figsize = (W * img_num, H)

        for batch_idx, (X, Y) in tqdm(enumerate(test_loader)):

            pred_Y = best_model.predict(X.to(p.DEVICE))
            img_sq_list = list()
            gt_sq_list = list()
            pr_sq_list = list()

            sq_info = None
            for square_idx in range(Y.shape[0]):
                n = batch_idx * 16 + square_idx
                sq_info = d.test_dataset.squares_info[n]

                image = X[square_idx].squeeze().cpu().numpy()
                gt_mask = Y[square_idx].squeeze().cpu().numpy().round()
                pr_mask = pred_Y[square_idx].squeeze().cpu().numpy().round()
                img_sq_list.append(image)
                gt_sq_list.append(gt_mask)
                pr_sq_list.append(pr_mask)

                iou_metric = smp.utils.functional.iou(torch.from_numpy(gt_mask), torch.from_numpy(pr_mask))
                iou_metric_list.append(iou_metric)

                # fig, ax = plt.subplots(1, img_num, figsize=figsize)
                # for c_idx in range(d.test_dataset.channels):
                #     ax[c_idx].imshow(image[c_idx], cmap='gray', vmin=0, vmax=1)
                #     ax[c_idx].title.set_text(f'# {c_idx}')

                # gt_full = gt_mask[0]
                # gt_full[gt_mask[1] == 1] = 2
                # pr_full = pr_mask[0]
                # pr_full[pr_mask[1] == 1] = 2
            
                # ax[p.channels + 0].imshow(gt_full, cmap='gray', vmin=0, vmax=d.test_dataset.classes_num)
                # ax[p.channels + 0].title.set_text('gt_mask')
                # ax[p.channels + 1].imshow(pr_full, cmap='gray', vmin=0, vmax=d.test_dataset.classes_num)
                # ax[p.channels + 1].title.set_text('pr_mask')
                # # pprint(sq)
                # # plt.show()
                # fn = f'{sq_info["fp"].split("/")[-1]}-{sq_info["w"]}_{sq_info["h"]}.png'
                # fig.savefig(osp.join(out_dir, fn))
                # plt.close(fig)

            restored_img = unsplit_image(img_sq_list, test_loader.dataset.squares, 'square_coords', test_loader.dataset.border)
            restored_gt = unsplit_image(gt_sq_list, test_loader.dataset.squares, 'square_coords', test_loader.dataset.border)
            restored_pr = unsplit_image(pr_sq_list, test_loader.dataset.squares, 'square_coords', test_loader.dataset.border)

            out_fn_pr = f'{sq_info["fp"].split("/")[-1]}.npy'
            with open(osp.join(out_dir, out_fn_pr), 'wb') as f:
                np.save(f, restored_pr)

            restored_gt_full = restored_gt[0]
            for idx in range(restored_gt.shape[0]):
                restored_gt_full[restored_gt[idx] == 1] = idx + 1
            restored_pr_full = restored_pr[0]
            for idx in range(restored_pr.shape[0]):
                restored_pr_full[restored_pr[idx] == 1] = idx + 1
            
            fig, ax = plt.subplots(1, img_num, figsize=figsize)
            for c_idx in range(d.test_dataset.channels):
                ax[c_idx].imshow(restored_img[c_idx], cmap='gray', vmin=0, vmax=1)
                ax[c_idx].title.set_text(f'# {c_idx}')
            ax[p.channels + 0].imshow(restored_gt_full, cmap='gray', vmin=0, vmax=d.test_dataset.classes_num)
            ax[p.channels + 0].title.set_text('gt_mask')
            ax[p.channels + 1].imshow(restored_pr_full, cmap='gray', vmin=0, vmax=d.test_dataset.classes_num)
            ax[p.channels + 1].title.set_text('pr_mask')
            # pprint(sq)
            # plt.show()
            fn = f'{sq_info["fp"].split("/")[-1]}.png'
            fig.savefig(osp.join(out_dir, 'full', fn))
            plt.close(fig)

            fig, ax = plt.subplots(1, 2, figsize=(W * 2, H))
            color_shift_red = (+100, -100, -100)
            color_shift_green = (-100, +100, -100)
            color_shift_blue = (-100, -100, +100)
            color_shift_violet = (+100, -100, +100)
            color_shift_yellow = (+100, +100, -100)

            img_gt = restored_img[0].copy()
            img_gt3 = np.stack([img_gt, img_gt, img_gt], axis=2) * 255
            red_idx = restored_gt[0] == 1
            for c_idx, c in enumerate(color_shift_red):
                img_gt3[..., c_idx][red_idx] += c
            green_idx = restored_gt[1] == 1
            for c_idx, c in enumerate(color_shift_green):
                img_gt3[..., c_idx][green_idx] += c
            if restored_gt.shape[0] > 2:
                blue_idx = restored_gt[2] == 1
                for c_idx, c in enumerate(color_shift_blue):
                    img_gt3[..., c_idx][blue_idx] += c
                yellow_idx = restored_gt[3] == 1
                for c_idx, c in enumerate(color_shift_yellow):
                    img_gt3[..., c_idx][yellow_idx] += c

            np.clip(img_gt3, 0, 255, out=img_gt3)
            img_gt3 = img_gt3.astype(np.uint8)
            ax[0].imshow(img_gt3)
            ax[0].title.set_text('img_gt')
            
            img_pr = restored_img[0].copy()
            img_pr3 = np.stack([img_pr, img_pr, img_pr], axis=2) * 255
            red_idx = restored_pr[0] == 1
            for c_idx, c in enumerate(color_shift_red):
                img_pr3[..., c_idx][red_idx] += c
            green_idx = restored_pr[1] == 1
            for c_idx, c in enumerate(color_shift_green):
                img_pr3[..., c_idx][green_idx] += c
            if restored_pr.shape[0] > 2:
                blue_idx = restored_pr[2] == 1
                for c_idx, c in enumerate(color_shift_blue):
                    img_pr3[..., c_idx][blue_idx] += c
                yellow_idx = restored_pr[3] == 1
                for c_idx, c in enumerate(color_shift_yellow):
                    img_pr3[..., c_idx][yellow_idx] += c

            np.clip(img_pr3, 0, 255, out=img_pr3)
            img_pr3 = img_pr3.astype(np.uint8)
            ax[1].imshow(img_pr3)
            ax[1].title.set_text('img_pr')
            
            fn = f'{sq_info["fp"].split("/")[-1]}.png'
            fig.savefig(osp.join(out_dir, 'compare', fn))
            plt.close(fig)
            
        print(f'IoU = {np.mean(iou_metric_list)}')

    print('DONE!')

