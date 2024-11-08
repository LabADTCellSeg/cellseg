from pathlib import Path

from types import SimpleNamespace
from pprint import pprint

import torch

from cellseg_exp import experiment
from cellseg_utils import prepare_data_from_params, get_str_timestamp

TORCH_HUB_DIR = '/storage0/pia/python/hub/'

torch.hub.set_dir(TORCH_HUB_DIR)

if __name__ == '__main__':
    draw = True
    test = False

    multiclass = True
    add_shadow_to_img = True
    square_a = 256
    border = 10
    contour_thickness = 4

    out_root = Path('out')
    dataset_root = Path('/storage0/pia/python/cellseg/')

    if test:
        run_clear_ml = False
        out_dir = out_root / 'test'
        shuffle = True
        ratio_train = 0.8
        ratio_val = 0.2
        images_num = 5
        max_epochs = 1
    else:
        run_clear_ml = False
        out_dir = out_root / 'LF1'
        shuffle = True
        ratio_train = 0.8
        ratio_val = 0.2
        images_num = None
        max_epochs = 50

    dataset_dir = dataset_root / Path('datasets/Cells_2.0_for_Ivan/masked_MSC')
    dir01 = dataset_dir / 'pics 2024-20240807T031703Z-001' / 'pics 2024'
    dir02 = dataset_dir / 'pics 2024-20240807T031703Z-002' / 'pics 2024'
    lf_dir = dir01 / 'LF1'
    resize_coef = 1

    dataset_dir = lf_dir
    # exp_class_dict = {'+2024-05-05-LF1-p6-sl2': 6,
    #                   '+2024-05-06-LF1p9-sl2': 9,
    #                   '+2024-05-06-LF1-p12': 12,
    #                   '+2024-05-07-LF1p15': 15,
    #                   '+2024-05-08-LF1p18sl2': 18,
    #                   '+2024-05-31-LF1-p22': 22}
    exp_class_dict = {'+2024-05-05-LF1-p6-sl2': 6,
                      '+2024-05-06-LF1p9-sl2': 6,
                      '+2024-05-06-LF1-p12': 12,
                      '+2024-05-07-LF1p15': 12,
                      '+2024-05-08-LF1p18sl2': 18,
                      '+2024-05-31-LF1-p22': 18}

    channels = 3
    if add_shadow_to_img:
        channels += 1  # shadow
    classes_num = len(set(exp_class_dict.values())) if multiclass else 1
    classes_num += 1  # border

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
        ratio_train=ratio_train,
        ratio_val=ratio_val,
        square_a=square_a,
        border=border,
        classes_num=classes_num,
        channels=channels,
        num_workers=4,
        batch_size=32,
        bce_weight=0.2,

        ENCODER=None,
        # 'resnet101',  # 'efficientnet-b2',  # 'timm-efficientnet-b8',  # 'efficientnet-b0'
        ENCODER_WEIGHTS='imagenet',
        ACTIVATION='sigmoid',  # could be None for logits or 'softmax2d' for multiclass segmentation
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu',

        max_epochs=max_epochs,
        lr_first=1e-2,
        lr_last=1e-5,
        scheduler_step_every_batch=True,
    )

    params = SimpleNamespace(**params)

    fp_data_list, aug_list, dataset_fn = prepare_data_from_params(params,
                                                                  shuffle=shuffle,
                                                                  max_workers=8)

    dataset_params = SimpleNamespace(fp_data_list=fp_data_list,
                                     aug_list=aug_list,
                                     dataset_fn=dataset_fn)

    # encoders = ['resnet34', 'resnet50', 'resnext50_32x4d', 'se_resnet50', 'resnet101', 'vgg19']
    # models = ['Unet', 'MAnet', 'FPN', 'DeepLabV3', 'DeepLabV3Plus']
    # models = ['DeepLabV3', 'DeepLabV3Plus']
    encoders = ['timm-efficientnet-b8']
    models = ['Unet']
    for encoder in encoders:
        params.ENCODER = encoder
        for model_name in models:
            params.model_name = model_name
            pprint(vars(params))

            log_dir = Path(out_dir) / f'{params.model_name}_{params.ENCODER}_{get_str_timestamp()}'
            experiment(run_clear_ml=run_clear_ml, p=params, d=dataset_params, log_dir=log_dir, draw=draw)
            torch.cuda.empty_cache()
