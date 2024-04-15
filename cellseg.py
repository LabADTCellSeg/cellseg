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

from cellseg_utils import (
    get_str_timestamp,
    get_squares,
    # unsplit_image,
    my_train_test_split,
    CellDataset,
    get_training_augmentation,
    get_validation_augmentation,
    get_preprocessing,
)

matplotlib.use('TkAgg')


def experiment(p):
    fn_list = [v for v in os.listdir(p.dataset_dir) if v.endswith('.nd2')]
    fn_list.sort()

    W = H = 5
    fp = osp.join(p.dataset_dir, fn_list[0])
    np_data = nd2.imread(fp)
    c, w, h = np_data.shape
    figsize = (W * c + 1, H)

    print(f'total images: {len(fn_list)}')
    print(f'image shape (c, w, h): {np_data.shape}')

    orig_size = (w, h)
    square_w, square_h = p.square_a, p.square_a
    square_size = (square_w, square_h)

    full_size, full_size_with_borders, squares = get_squares(orig_size,
                                                             square_size,
                                                             p.border)

    print(f'orig_size: {orig_size}')
    print(f'full_size: {full_size}')
    print(f'full_size_with_borders: {full_size_with_borders}')
    # pprint(core_square_sizes)
    print(f'squares: {len(squares)}')

    pprint(squares[:2])
    print('...')
    pprint(squares[-2:])

    images_fps = list()
    masks_fps = list()
    for fn in fn_list[:]:
        images_fps.append(osp.join(p.dataset_dir, fn))
        masks_fps.append(osp.join(p.masks_dir, f'{fn}mask.png'))
    images_fps = np.array(images_fps)
    masks_fps = np.array(masks_fps)

    run_clear_ml = False

    timestamp = get_str_timestamp()

    log_dir = f'out/MSC/{p.model_name}_{p.ENCODER}_{timestamp}'
    os.makedirs(log_dir)

    ans = my_train_test_split(images_fps, masks_fps, p.ratio_train, p.ratio_val)
    X_train, X_val, X_test, y_train, y_val, y_test = ans
    print(X_train.shape, X_val.shape, X_test.shape)

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(p.ENCODER, p.ENCODER_WEIGHTS)
    preprocessing_fn = None
    preprocessing = get_preprocessing(preprocessing_fn)

    if p.max_epochs != 0:
        train_dataset = CellDataset(X_train, y_train, squares,
                                    p.border, p.channels, p.classes, full_size,
                                    augmentation=get_training_augmentation(),
                                    preprocessing=preprocessing)

        valid_dataset = CellDataset(X_val, y_val, squares,
                                    p.border, p.channels, p.classes, full_size,
                                    augmentation=get_training_augmentation(),
                                    preprocessing=preprocessing)

    test_dataset = CellDataset(X_test, y_test, squares,
                               p.border, p.channels, p.classes, full_size,
                               augmentation=get_validation_augmentation(),
                               preprocessing=preprocessing)
    if p.max_epochs != 0:
        train_loader = DataLoader(train_dataset, batch_size=p.batch_size,
                                  shuffle=True, num_workers=p.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=p.batch_size,
                                  shuffle=False, num_workers=p.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False)

    # create segmentation model with pretrained encoder
    if p.model_load_fp is None:
        model_func = getattr(importlib.import_module('segmentation_models_pytorch'),
                             p.model_name)
        model = model_func(
            encoder_name=p.ENCODER,
            encoder_weights=p.ENCODER_WEIGHTS,
            in_channels=p.IN_CHANNELS,
            classes=p.classes,
            activation=p.ACTIVATION,
        )
        print('created new')
    else:
        model = torch.load(p.model_load_fp)
        print(f'loaded {p.model_load_fp}')

    summary(model)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.losses.DiceLoss(mode='multilabel')
    loss.__name__ = 'dice'
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=p.lr_first),
    ])

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
        task.connect(p)
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
        print(f"{' '.join(k.split('_')).title()}: {v:.2f}")

    # visualize results of best saved model
    iou_metric_list = list()

    out_dir = os.path.join(log_dir, 'results')
    os.makedirs(out_dir, exist_ok=True)
    for d in ['image', 'gt', 'pr', 'compare']:
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    for n in tqdm(range(len(test_dataset))):
        image, gt_mask = test_dataset[n]
        gt_mask = gt_mask

        x_tensor = torch.from_numpy(image).to(p.DEVICE).unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())

        iou_metric = smp.utils.functional.iou(torch.from_numpy(gt_mask), torch.from_numpy(pr_mask))
        iou_metric_list.append(iou_metric)

        # print(iou_metric)

        fig, ax = plt.subplots(1, p.channels + 2, figsize=figsize)
        for c_idx in range(c):
            ax[c_idx].imshow(image[c_idx], cmap='gray', vmin=0, vmax=1)
            ax[c_idx].title.set_text(f'# {c_idx}')
        ax[2].imshow(gt_mask[0] + gt_mask[1], cmap='gray', vmin=0, vmax=gt_mask.shape[0])
        ax[2].title.set_text('gt_mask')
        ax[3].imshow(pr_mask[0] + pr_mask[1], cmap='gray', vmin=0, vmax=pr_mask.shape[0])
        ax[3].title.set_text('pr_mask')
        # pprint(sq)
        # plt.show()
        sq_info = test_dataset.squares_info[n]
        fn = f'{sq_info["fp"].split("/")[-1]}-{sq_info["w"]}_{sq_info["h"]}.png'
        fig.savefig(osp.join(out_dir, fn))

    print('DONE!')
    print(f'IoU = {np.mean(iou_metric_list)}')
