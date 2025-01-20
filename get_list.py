from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map
import shutil

from cellseg_utils import get_all_fp_data
from cellseg_get_stats_utils import (
    get_cell_statistics,
    draw_ellipses
)

CASCADE = False
max_workers = 8

if CASCADE:
    server_name = 'CASCADE'
    root_dir = '/storage0/pia/python/cellseg/'
else:
    server_name = 'seth'
    root_dir = '.'

if __name__ == '__main__':
    multiclass = False
    add_shadow_to_img = False

    root_dir = Path(root_dir)

    dataset_dir = root_dir / Path('datasets/Cells_2.0_for_Ivan/masked_MSC')
    dir01 = dataset_dir / 'pics 2024-20240807T031703Z-001' / 'pics 2024'
    dir02 = dataset_dir / 'pics 2024-20240807T031703Z-002' / 'pics 2024'
    lf_dir = dir01 / 'LF1'
    wj_msc_dir = dir02 / 'WJ-MSC-P57'

    # dataset_dir = lf_dir
    # exp_class_dict = {'+2024-05-05-LF1-p6-sl2': 1,
    #                   '+2024-05-06-LF1p9-sl2': 1,
    #                   '+2024-05-06-LF1-p12': 2,
    #                   '+2024-05-07-LF1p15': 2,
    #                   '+2024-05-08-LF1p18sl2': 3,
    #                   '+2024-05-31-LF1-p22': 3}
    # channels= ['r', 'g', 'b']

    dataset_dir = wj_msc_dir
    exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 1,
                      '2024-05-03-wj-MSC-P57p5': 1,
                      '2024-05-01-wj-MSC-P57p7': 2,
                      '2024-05-04-wj-MSC-P57p9-sl2': 2,
                      '2024-05-02-wj-MSC-P57p11': 3,
                      '2024-05-03-wj-MSC-P57p13': 3,
                      '2024-05-02-wj-MSC-P57p15sl2': 3}
    channels = ['b']

    model_results_dir = Path('out_CASCADE/out/WJ-MSC-P57/DeepLabV3Plus_timm-efficientnet-b0_20241119_172741/test_results')
    # model_results_dir = Path('out_CASCADE/out/WJ-MSC-P57/MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611/results')

    out_dir = model_results_dir / 'stats_results'
    predicted_masks_dir = model_results_dir / 'predicted_masks'

    all_fp_data = get_all_fp_data(dataset_dir, exp_class_dict)
    all_fp_data = sorted(all_fp_data, key=lambda d: d['idx'])

    exp = 'WJ-MSC-P57'
    samples_to_analyze = list()
    for sample_data in all_fp_data:
        pred_mask_fp_list = [
            str(v) for v in predicted_masks_dir.glob(f'{sample_data["idx"]}_*')]
        pred_mask_fp_list.sort()
        if len(pred_mask_fp_list) > 0:
            samples_to_analyze.append(sample_data)
    
    info_list = list()
    for sample_data in samples_to_analyze:
        idx = sample_data['idx']
        image_fp = sample_data['b_fp']
        mask_fp = sample_data['mask_fp']

        split_fp = str(image_fp).split('/')
        exp_class_dir = split_fp[-2]        
        info_list.append([exp_class_dir, image_fp.name])
        
    info_df = pd.DataFrame(info_list, columns=['dir', 'filename'])
    info_df.to_csv(out_dir / 'file_list.csv', index=0)
