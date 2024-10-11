import os
from types import SimpleNamespace
from pprint import pprint

import torch

from cellseg import experiment
from cellseg_utils import prepare_data, get_str_timestamp

if __name__ == '__main__':
    draw = True
    test = True
    
    if test:
        run_clear_ml = False
        out_dir = 'out/test'
        shuffle = True
        ratio_train = 0.0
        ratio_val = 0.0
        # ratio_train = 0.6
        # ratio_val = 0.2
        images_num = None
        max_epochs = 0
    else:
        run_clear_ml = True
        # run_clear_ml = False

        # out_dir = 'out/MSC'
        out_dir = 'out/MSC_upd_mc'
        shuffle = True
        ratio_train = 0.6
        ratio_val = 0.2
        images_num = None
        max_epochs = 200
    dataset_dir='datasets/MSC/30_04_2023-LF1-P6-P21'
    
    params = dict(
        model_name='Unet',
        # model_load_fp='out/MSC_filtered_mc/Unet_timm-efficientnet-b0_20240425_160139/best_model.pth',
        # model_load_fp=os.path.join('out', 'MSC', 'Unet_timm-efficientnet-b6_20240417_144310', 'best_model.pth'),
        # model_load_fp=os.path.join('out', 'MSC_upd', 'Unet_timm-efficientnet-b6_2c_20240509_105457', 'best_model.pth'),
        model_load_fp=os.path.join('out', 'MSC_upd_mc', 'Unet_timm-efficientnet-b6_4c_20240509_230818', 'best_model.pth'),

        # model_load_fp=None,
        model_load_full=True,

        dataset_dir=dataset_dir,
        masks_dir='datasets/MSC/30_04_2023-LF1-P6-P21_masks_upd',
        ratio_train=ratio_train,
        ratio_val=ratio_val,
        
        square_a=256,
        border=64,
        classes_num=1+3,  #border + passages
        channels=2+2,  # nd2 + first_segmentation
        num_workers=1,
        batch_size=2,
        allowed_markers=None,
        bce_weight=0.1,
        # allowed_markers=['Ki67'],

        ENCODER='timm-efficientnet-b6',  # 'resnet101',  # 'efficientnet-b2',  # 'timm-efficientnet-b8',  # 'efficientnet-b0'
        ENCODER_WEIGHTS='imagenet',
        ACTIVATION='sigmoid',  # could be None for logits or 'softmax2d' for multiclass segmentation
        DEVICE='cuda' if torch.cuda.is_available() else 'cpu',

        max_epochs=max_epochs,
        lr_first=1e-3,
        lr_last=1e-4,
    )

    params = SimpleNamespace(**params)

    # encoders = ['resnet34', 'resnet50', 'resnext50_32x4d', 'se_resnet50', 'resnet101', 'vgg19']
    # models = ['Unet', 'MAnet', 'FPN', 'DeepLabV3', 'DeepLabV3Plus']
    # models = ['DeepLabV3', 'DeepLabV3Plus']

    encoders = ['timm-efficientnet-b6']
    models = ['Unet']
    # markers = ['Ki67', 'H3K9-3mc', 'HP1a', 'HP1b', 'HP1g']
    markers = [None]

    for encoder in encoders:
        params.ENCODER = encoder
        for model_name in models:
            params.model_name = model_name
            for m in markers:
                if m is None:
                    params.allowed_markers = None
                else:
                    params.allowed_markers = [m]
                
                train_dataset, valid_dataset, test_dataset = prepare_data(params,
                                                                          images_num=images_num,
                                                                          shuffle=shuffle,
                                                                          classes_groups_num=params.classes_num-1,
                                                                          allowed_markers=params.allowed_markers)
                datasets = SimpleNamespace(train_dataset=train_dataset,
                                           valid_dataset=valid_dataset,
                                           test_dataset=test_dataset)

                pprint(vars(params))

                log_dir = os.path.join(out_dir, f'{params.model_name}_{params.ENCODER}_{params.classes_num}c')
                if params.allowed_markers is not None:
                    log_marker_str = ''
                    for m in params.allowed_markers:
                        log_marker_str += f'_{m}'
                    log_dir += log_marker_str
                log_dir += f'_{get_str_timestamp()}'
                experiment(run_clear_ml=run_clear_ml, p=params, d=datasets, log_dir=log_dir, draw=draw)
                torch.cuda.empty_cache()
