import os
from pathlib import Path

from types import SimpleNamespace
from pprint import pprint

import torch

from cellseg_exp import experiment
from cellseg_utils import prepare_data, get_str_timestamp

if __name__ == '__main__':
    draw = True
    test = False
    
    if test:
        run_clear_ml = False
        out_dir = 'out/test'
        shuffle = False
        ratio_train = 0.0
        ratio_val = 0.0
        # ratio_train = 0.6
        # ratio_val = 0.2
        # images_num = 20
        images_num = None
        max_epochs = 0
    else:
        run_clear_ml = True
        out_dir = 'out/LF1'
        shuffle = True
        ratio_train = 0.6
        ratio_val = 0.2
        images_num = 10
        max_epochs = 50

    root_dir = Path('datasets/Cells_2.0_for_Ivan/masked_MSC')
    dir01 = root_dir / 'pics 2024-20240807T031703Z-001' / 'pics 2024'
    dir02 = root_dir / 'pics 2024-20240807T031703Z-002' / 'pics 2024'
    lf_dir = dir01 / 'LF1'

    dataset_dir = lf_dir
    exp_class_dict = {'+2024-05-05-LF1-p6-sl2': 6,
                      '+2024-05-06-LF1-p12': 12,
                      '+2024-05-06-LF1p9-sl2': 9,
                      '+2024-05-07-LF1p15': 15,
                      '+2024-05-08-LF1p18sl2': 18,
                      '+2024-05-31-LF1-p22': 22
                      }

    params = dict(
        model_name='Unet',
        # model_load_fp='out/MSC_filtered_mc/Unet_timm-efficientnet-b0_20240425_160139/best_model.pth',
        # model_load_fp=os.path.join('out', 'MSC', 'Unet_timm-efficientnet-b6_20240417_144310', 'best_model.pth'),
        model_load_fp=None,
        model_load_full=True,

        dataset_dir=dataset_dir,
        exp_class_dict=exp_class_dict,
        ratio_train=ratio_train,
        ratio_val=ratio_val,
        
        square_a=None,
        border=None,
        classes_num=1+1,
        channels=4,
        num_workers=1,
        batch_size=2,
        bce_weight=0.1,

        ENCODER='timm-efficientnet-b0',  # 'resnet101',  # 'efficientnet-b2',  # 'timm-efficientnet-b8',  # 'efficientnet-b0'
        ENCODER_WEIGHTS='imagenet',
        ACTIVATION='sigmoid',  # could be None for logits or 'softmax2d' for multiclass segmentation
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu',

        max_epochs=max_epochs,
        lr_first=1e-5,
        lr_last=1e-7,
    )

    params = SimpleNamespace(**params)

    train_dataset, valid_dataset, test_dataset = prepare_data(params, images_num=images_num, shuffle=shuffle)
    
    datasets = SimpleNamespace(train_dataset=train_dataset,
                               valid_dataset=valid_dataset,
                               test_dataset=test_dataset)

    # encoders = ['resnet34', 'resnet50', 'resnext50_32x4d', 'se_resnet50', 'resnet101', 'vgg19']
    # models = ['Unet', 'MAnet', 'FPN', 'DeepLabV3', 'DeepLabV3Plus']
    # models = ['DeepLabV3', 'DeepLabV3Plus']

    encoders = ['timm-efficientnet-b6']
    models = ['Unet']
    for encoder in encoders:
        params.ENCODER = encoder
        for model_name in models:
            params.model_name = model_name
            pprint(vars(params))

            log_dir = os.path.join(out_dir, f'{params.model_name}_{params.ENCODER}_{get_str_timestamp()}')
            experiment(run_clear_ml=run_clear_ml, p=params, d=datasets, log_dir=log_dir, draw=draw)
            torch.cuda.empty_cache()
