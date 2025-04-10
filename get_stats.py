# This script calculates cell statistics from segmentation results and saves the results as CSV and image files.

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

from cellseg_config import *


if __name__ == '__main__':
    max_workers = 8  # Maximum number of parallel workers (typically equal to CPU cores)
    exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 1,
                      '2024-05-03-wj-MSC-P57p5': 1,
                      '2024-05-01-wj-MSC-P57p7': 2,
                      '2024-05-04-wj-MSC-P57p9-sl2': 2,
                      '2024-05-02-wj-MSC-P57p11': 3,
                      '2024-05-03-wj-MSC-P57p13': 3,
                      '2024-05-02-wj-MSC-P57p15sl2': 3}

    # Set up directories using configuration values
    # model_results_dir = 'out/WJ-MSC-P57/MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611'
    exp_dir = Path(exp_dir)
    dataset_dir = Path(dataset_dir)
    model_results_dir = Path(model_results_dir)

    out_dir = model_results_dir / 'stats_results'
    predicted_masks_dir = model_results_dir / 'predicted_masks'

    res_csv_stat_dir = out_dir / 'csv_stat'
    res_csv_stat_dir.mkdir(exist_ok=True, parents=True)

    all_fp_data = get_all_fp_data(dataset_dir, exp_class_dict)
    # Sort file pointer data by index
    all_fp_data = sorted(all_fp_data, key=lambda d: d['idx'])

    # Define color shifts for visualization
    color_shift_red = (+100, -100, -100)
    color_shift_green = (-200, +200, -200)
    color_shift_blue = (-100, -100, +100)
    color_shift_yellow = (+100, +100, -100)
    color_shift_white = (+255, +255, +255)

    color_shift_list = [color_shift_red,
                        color_shift_blue,
                        color_shift_yellow]

    # Set erosion and dilation parameters
    er = 20
    dil = er

    # Select samples that have predicted masks available
    samples_to_analyze = list()
    for sample_data in all_fp_data:
        pred_mask_fp_list = [
            str(v) for v in predicted_masks_dir.glob(f'{sample_data["idx"]}_*')]
        pred_mask_fp_list.sort()
        if len(pred_mask_fp_list) > 0:
            samples_to_analyze.append([sample_data,
                                       pred_mask_fp_list])

    # Parameters for result generation
    num_samples = None
    rewrite_existed = True

    def process(args):
        # Process a single sample: load image, mask and predicted masks, compute statistics and save outputs
        sample_data, pred_mask_fp_list = args

        idx = sample_data['idx']
        image_fp = sample_data['b_fp']
        mask_fp = sample_data['mask_fp']

        split_fp = str(image_fp).split('/')
        exp_class_dir = split_fp[-2]
        exp_class_value = exp_class_dict[exp_class_dir]

        exp_out_dir = out_dir / exp_class_dir

        exp_out_dir.mkdir(exist_ok=True, parents=True)
        shutil.copy(mask_fp, exp_out_dir / mask_fp.name)

        for rgb_fp in ['b_fp', 'r_fp', 'g_fp']:
            src_fp = sample_data.get(rgb_fp)
            if src_fp is not None:
                dst_fn = src_fp.name
                shutil.copy(src_fp, exp_out_dir / dst_fn)

        res_csv_fp = (exp_out_dir / f'{idx}result').with_suffix('.csv')
        res_img_fp = (exp_out_dir / f'{idx}result').with_suffix('.png')

        # Process and generate predicted masks if result files are missing or rewriting is allowed
        if not res_csv_fp.exists() or not res_img_fp.exists() or rewrite_existed:
            img = np.asarray(Image.open(image_fp))
            mask = np.asarray(Image.open(mask_fp))
            pred_masks = list()
            for pred_mask_idx, pred_mask_fp in enumerate(pred_mask_fp_list):
                m = np.asarray(Image.open(pred_mask_fp))
                m = cv2.resize(m, (img.shape[1], img.shape[0]))
                pred_masks.append(m)

                dst_fn = f'{idx}m_pred{pred_mask_idx}{Path(pred_mask_fp).suffix}'
                Image.fromarray(m.astype(np.uint8)).save(exp_out_dir / dst_fn)

            pred_masks = np.stack(pred_masks, axis=-1)

        # Calculate cell statistics and save CSV results
        if rewrite_existed or not res_csv_fp.exists():
            passage_mask = None
            if pred_masks.shape[-1] > 2:
                passage_mask = pred_masks[..., :-1].transpose(2, 0, 1)
                passage_mask[passage_mask > 0] = 1

            main_mask = np.zeros((pred_masks.shape[0], pred_masks.shape[1]))
            for i in range(pred_masks.shape[-1]-1):
                main_mask[pred_masks[..., i] > 0] = 1

            contour_mask = pred_masks[..., -1].copy()
            contour_mask[contour_mask > 0] = 1
            main_mask[contour_mask == 1] = 0

            if er != 0:
                kernel = np.ones((er, er), np.uint8)
                main_mask = cv2.erode(main_mask, kernel)
            if dil != 0:
                kernel = np.ones((dil, dil), np.uint8)
                main_mask = cv2.dilate(main_mask, kernel)

            p = exp_class_value
            pgr = exp_class_value
            marker = 'b'
            n = idx
            statistics_df = get_cell_statistics(
                main_mask, exp, exp_class_dir, p, pgr, marker, n, passage_mask=passage_mask)
            statistics_df.to_csv(res_csv_fp, index=0)
        else:
            statistics_df = pd.read_csv(res_csv_fp, index_col=0)

        # Generate and save the result image overlay
        if rewrite_existed or not res_img_fp.exists():
            contours_image = draw_ellipses(statistics_df, target_size=(
                img.shape[0], img.shape[1]), hd_max=50, thickness=3)

            result_matrix = img.copy().astype(np.int16)

            color_shift_idx_list = list()
            for i in range(pred_masks.shape[-1]-1):
                color_shift_idx_list.append([color_shift_list[i],
                                             pred_masks[..., i] == 255])
            color_shift_idx_list.append([color_shift_green,
                                         pred_masks[..., -1] == 255])
            color_shift_idx_list.append([color_shift_white,
                                         contours_image == 1])

            for color_shift, color_idx in color_shift_idx_list:
                for c_idx, c in enumerate(color_shift):
                    result_matrix[..., c_idx][color_idx] += c

            np.clip(result_matrix, 0, 255, out=result_matrix)
            result_matrix = result_matrix.astype(np.uint8)

            result_img = Image.fromarray(result_matrix.astype(np.uint8))
            result_img.save(res_img_fp)

    # Parallel processing of samples
    process_map(process, samples_to_analyze[:num_samples], max_workers=max_workers, chunksize=1)

    # Aggregate results: concatenate CSV files and save full statistics
    csv_list = list(out_dir.rglob('*.csv'))
    csv_list.sort()
    csv_list = list(filter(lambda x: 'csv_stat' not in str(x), csv_list))

    pd_list = list()
    for csv_fp in csv_list:
        pd_list.append(pd.read_csv(csv_fp))
    result_pd = pd.concat(pd_list).reset_index(drop=True)
    # result_pd['Pred_true'] = result_pd['Pred_PGr'] == result_pd['PGr']
    result_pd.to_csv(res_csv_stat_dir / 'result_all.csv', index=0)

    # Generate statistics grouped by marker
    Marker_unique = result_pd['Marker'].unique().tolist()
    Marker_unique.sort()
    for m in Marker_unique:
        m_pd = result_pd[result_pd['Marker'] == m].reset_index(drop=True)
        m_pd = m_pd.sort_values(['P'])
        m_pd.to_csv(res_csv_stat_dir / f'result_{m}.csv', index=0)

    # Compute counts and mean areas for each experiment and sample group
    Exp_list = []
    Exp_dir_list = []
    P_list = []
    PGr_list = []
    Marker_list = []
    N_list = []
    count_list = []
    mean_area_list = []
    pred_acc_list = []

    Exp_unique = result_pd['Exp'].unique().tolist()
    Exp_unique.sort()
    N_unique = result_pd['N'].unique().tolist()
    N_unique.sort()
    for e in Exp_unique:
        e_pd = result_pd[result_pd['Exp'] == e]
        for n in N_unique:
            n_pd = e_pd[e_pd['N'] == n]
            count = n_pd.reset_index(drop=True).shape[0]
            mean_area = n_pd['Area'].mean()
            # pred_acc = n_pd['Pred_true'].mean()
            if count != 0:
                Exp_list.append(n_pd['Exp'].iloc[0])
                Exp_dir_list.append(n_pd['Exp_dir'].iloc[0])
                P_list.append(n_pd['P'].iloc[0])
                PGr_list.append(n_pd['PGr'].iloc[0])
                Marker_list.append(n_pd['Marker'].iloc[0])
                N_list.append(n_pd['N'].iloc[0])
                count_list.append(count)
                mean_area_list.append(mean_area)
                # pred_acc_list.append(pred_acc)
    count_pd = pd.DataFrame({
        'Exp': Exp_list,
        'Exp_dir': Exp_dir_list,
        'P': P_list,
        'PGr': PGr_list,
        'Marker': Marker_list,
        'N': N_list,
        'count': count_list,
        'mean area': mean_area_list,
        # 'pred acc': pred_accs,
    })
    count_pd.to_csv(res_csv_stat_dir / 'result_all_count_area.csv', index=0)
    count_pd
