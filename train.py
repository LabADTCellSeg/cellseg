# This script sets up and runs the training process for the segmentation model.

from pathlib import Path
from types import SimpleNamespace
from pprint import pprint

import torch

from cellseg_exp import experiment
from cellseg_utils import get_str_timestamp
from cellseg_dataset import prepare_data_from_params


CASCADE = False
run_clear_ml = True
test = True

if CASCADE:
    server_name = 'CASCADE'
    TORCH_HUB_DIR = '/storage0/pia/python/hub/'
    torch.hub.set_dir(TORCH_HUB_DIR)
    root_dir = '/storage0/pia/python/cellseg/'
    run_clear_ml = False
else:
    server_name = 'seth'
    root_dir = '.'


if __name__ == '__main__':
    draw = True
    multiclass = True
    add_shadow_to_img = False
    square_a = 256
    border = 10
    contour_thickness = 2

    out_root = Path('out')
    root_dir = Path(root_dir)

    # Setup dataset directories and experiment-specific parameters
    dataset_dir = root_dir / Path('datasets/Cells_2.0_for_Ivan/masked_MSC')
    dir01 = dataset_dir / 'pics 2024-20240807T031703Z-001' / 'pics 2024'
    dir02 = dataset_dir / 'pics 2024-20240807T031703Z-002' / 'pics 2024'
    lf_dir = dir01 / 'LF1'
    wj_msc_dir = dir02 / 'WJ-MSC-P57'

    resize_coef = 1

    # Uncomment below for LF1 experiment configuration if needed
    # dataset_dir = lf_dir
    # exp_class_dict = {'+2024-05-05-LF1-p6-sl2': 6,
    #                   '+2024-05-06-LF1p9-sl2': 6,
    #                   '+2024-05-06-LF1-p12': 12,
    #                   '+2024-05-07-LF1p15': 12,
    #                   '+2024-05-08-LF1p18sl2': 18,
    #                   '+2024-05-31-LF1-p22': 18}
    # channels= ['r', 'g', 'b']

    # Set experiment to WJ-MSC-P57 configuration
    dataset_dir = wj_msc_dir
    exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 3,
                      '2024-05-03-wj-MSC-P57p5': 3,
                      '2024-05-01-wj-MSC-P57p7': 7,
                      '2024-05-04-wj-MSC-P57p9-sl2': 7,
                      '2024-05-02-wj-MSC-P57p11': 11,
                      '2024-05-03-wj-MSC-P57p13': 11,
                      '2024-05-02-wj-MSC-P57p15sl2': 11}
    classes = [3, 7, 11]
    channels = ['b']

    if test:
        run_clear_ml = False
        out_dir = out_root / 'test'
        shuffle = True
        images_num = 5
        max_epochs = 1
    else:
        out_dir = out_root / dataset_dir.stem
        shuffle = True
        images_num = None
        max_epochs = 50

    channels_num = len(channels)
    if add_shadow_to_img:
        channels_num += 1  # Add extra channel for shadow
    classes_num = len(set(exp_class_dict.values())) if multiclass else 1
    classes_num += 1  # Include border class

    square_a = square_a - (border * 2)
    params = dict(
        model_name=None,
        model_load_fp=None,
        # model_load_fp=Path('out/LF1') / 'Unet_timm-efficientnet-b6_20241012_154141' / 'best_model.pth',
        model_load_full=True,

        dataset_dir=dataset_dir,
        exp_class_dict=exp_class_dict,
        images_num=images_num,
        multiclass=multiclass,
        add_shadow_to_img=add_shadow_to_img,
        contour_thickness=contour_thickness,
        ratio_train=0.8,
        ratio_val=0.2,
        square_a=square_a,
        border=border,
        classes_num=classes_num,
        channels=channels,
        channels_num=channels_num,
        num_workers=4,
        batch_size=4,
        bce_weight=0.0,
        focal_alpha=0.5,
        focal_gamma=1.5,

        ENCODER=None,
        # 'resnet101',  # 'efficientnet-b2',  # 'timm-efficientnet-b8',  # 'efficientnet-b0'
        ENCODER_WEIGHTS='imagenet',
        ACTIVATION='sigmoid',  # can be None for logits or 'softmax2d' for multiclass segmentation
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu',

        max_epochs=max_epochs,
        lr_first=1e-3,
        lr_last=1e-5,
        scheduler_step_every_batch=True,
    )

    params = SimpleNamespace(**params)

    # Prepare data for training and validation
    fp_data_list, aug_list, dataset_fn, dataset_test_fn = prepare_data_from_params(params, classes,
                                                                                   shuffle=shuffle,
                                                                                   max_workers=8)

    dataset_params = SimpleNamespace(fp_data_list=fp_data_list,
                                     aug_list=aug_list,
                                     dataset_fn=dataset_fn)

    # Define the encoder and model to be use
    # models = ['Unet', 'MAnet', 'FPN', 'DeepLabV3', 'DeepLabV3Plus']
    # models = ['DeepLabV3', 'DeepLabV3Plus']
    encoders = ['timm-efficientnet-b0']
    models = ['DeepLabV3Plus']
    for encoder in encoders:
        params.ENCODER = encoder
        for model_name in models:
            params.model_name = model_name
            pprint(vars(params))

            log_dir = Path(out_dir) / f'{params.model_name}_{params.ENCODER}_{get_str_timestamp()}'
            if multiclass:
                log_dir = log_dir.with_name('MC_' + log_dir.name)
            experiment(run_clear_ml=run_clear_ml, p=params, d=dataset_params, log_dir=log_dir, draw=draw)
            torch.cuda.empty_cache()
