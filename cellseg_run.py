import os
from types import SimpleNamespace
import torch
from pprint import pprint

from cellseg import experiment

if __name__ == '__main__':
    params = dict(
        model_name='Unet',
        model_load_fp=None,  # 'out/MSC/Unet_20240415_154121/best_model.pth'
        # model_load_fp='out/MSC/Unet_20240415_161246/best_model.pth',  # 'out/MSC/Unet_20240415_154121/best_model.pth'

        dataset_dir='datasets/MSC/30_04_2023-LF1-P6-P21',
        masks_dir='datasets/MSC/30_04_2023-LF1-P6-P21_masks',
        ratio_train=0.6,
        ratio_val=0.2,

        square_a=256,
        border=32,
        classes=2,
        channels=2,
        num_workers=1,
        batch_size=4,

        ENCODER='efficientnet-b2',  # 'timm-efficientnet-b8',  # 'efficientnet-b0', 'se_resnext50_32x4d'
        # ENCODER='timm-efficientnet-b8',  # 'timm-efficientnet-b8',  # 'efficientnet-b0', 'se_resnext50_32x4d'
        ENCODER_WEIGHTS='imagenet',
        IN_CHANNELS=2,
        ACTIVATION='sigmoid',  # could be None for logits or 'softmax2d' for multiclass segmentation
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu',

        max_epochs=50,
        lr_first=1e-3,
        lr_last=1e-4,
    )
    pprint(params)

    experiment(SimpleNamespace(**params))
