from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from cellseg_utils import get_all_fp_data
from cellseg_get_stats_utils import (
    get_cell_statistics,
    draw_ellipses
)

CASCADE = False

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
    # exp_class_dict = {'+2024-05-05-LF1-p6-sl2': 6,
    #                   '+2024-05-06-LF1p9-sl2': 6,
    #                   '+2024-05-06-LF1-p12': 12,
    #                   '+2024-05-07-LF1p15': 12,
    #                   '+2024-05-08-LF1p18sl2': 18,
    #                   '+2024-05-31-LF1-p22': 18}
    # channels= ['r', 'g', 'b']

    dataset_dir = wj_msc_dir
    exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 3,
                      '2024-05-03-wj-MSC-P57p5': 3,
                      '2024-05-01-wj-MSC-P57p7': 7,
                      '2024-05-04-wj-MSC-P57p9-sl2': 7,
                      '2024-05-02-wj-MSC-P57p11': 11,
                      '2024-05-03-wj-MSC-P57p13': 11,
                      '2024-05-02-wj-MSC-P57p15sl2': 11}
    channels = ['b']

    model_results_dir = Path(
        'out_CASCADE/out/WJ-MSC-P57/DeepLabV3Plus_timm-efficientnet-b0_20241119_172741')
    predicted_masks_dir = model_results_dir / 'test_results' / 'predicted_masks'
    res_csv_dir = predicted_masks_dir / 'CSV'
    res_csv_stat_dir = predicted_masks_dir / 'CSV_STAT'
    res_img_dir = predicted_masks_dir / 'CSV_IMG'
    res_csv_dir.mkdir(exist_ok=True)
    res_csv_stat_dir.mkdir(exist_ok=True)
    res_img_dir.mkdir(exist_ok=True)

    all_fp_data = get_all_fp_data(dataset_dir, exp_class_dict)
    all_fp_data = sorted(all_fp_data, key=lambda d: d['idx'])

    color_shift_red = (+100, -100, -100)
    color_shift_green = (-200, +200, -200)
    color_shift_blue = (-100, -100, +100)
    color_shift_white = (+255, +255, +255)

    er = 20
    dil = er

    exp = 'WJ-MSC-P57'
    samples_to_analyze = list()
    for sample_data in all_fp_data:
        pred_mask_fp_list = [str(v) for v in predicted_masks_dir.glob(f'{sample_data["idx"]}_*')]
        pred_mask_fp_list.sort()
        if len(pred_mask_fp_list) > 0:
            samples_to_analyze.append([sample_data['idx'],
                                       sample_data['b_fp'],
                                       sample_data['mask_fp'],
                                       pred_mask_fp_list])
    
    # result tables and images
    num_samples = None
    rewrite_existed = False
    for idx, image_fp, mask_fp, pred_mask_fp_list in tqdm(samples_to_analyze[:num_samples]):
        exp_class_dir = str(image_fp).split('/')[-2]
        exp_class_value = exp_class_dict[exp_class_dir]
        
        res_csv_fp = (res_csv_dir / idx).with_suffix('.csv')
        res_img_fp = (res_img_dir / idx).with_suffix('.png')
        
        if not res_csv_fp.exists() or not res_img_fp.exists() or rewrite_existed:
            img = np.asarray(Image.open(image_fp))
            mask = np.asarray(Image.open(mask_fp))
            pred_masks = list()
            for v in pred_mask_fp_list:
                m = np.asarray(Image.open(v))
                m = cv2.resize(m, (img.shape[1], img.shape[0]))
                pred_masks.append(m)
            pred_masks = np.stack(pred_masks, axis=-1)

        if rewrite_existed or not res_csv_fp.exists():
            passage_mask = None
            main_mask = pred_masks[..., 0].copy()
            contour_mask = pred_masks[..., -1].copy()
            main_mask[contour_mask == 255] = 0

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
                main_mask, exp, p, pgr, marker, n, passage_mask=passage_mask)
            statistics_df.to_csv(res_csv_fp, index=0)
        else:
            statistics_df = pd.read_csv(res_csv_fp, index_col=0)

        if rewrite_existed or not res_img_fp.exists():
            contours_image = draw_ellipses(statistics_df, target_size=(img.shape[0], img.shape[1]), hd_max=50, thickness=3)

            result_matrix = img.copy().astype(np.int16)
            red_idx = pred_masks[..., 0] == 255
            green_idx = pred_masks[..., -1] == 255
            # blue_idx = pred_masks[..., 1] == 255

            white_idx = contours_image == 1
            # blue_idx = contour_mask == 1

            pallete = [[color_shift_red, red_idx],
                        [color_shift_green, green_idx],
                        # [color_shift_blue, blue_idx],
                        [color_shift_white, white_idx],
                        ]
            for color_shift, color_idx in pallete:
                for c_idx, c in enumerate(color_shift):
                    result_matrix[..., c_idx][color_idx] += c

            np.clip(result_matrix, 0, 255, out=result_matrix)
            result_matrix = result_matrix.astype(np.uint8)

            result_img = Image.fromarray(result_matrix.astype(np.uint8))
            result_img.save(res_img_fp)

    # results for all samples
    csv_list = list(res_csv_dir.glob('*.csv'))
    csv_list.sort()
    pd_list = list()
    for csv_fp in csv_list:
        pd_list.append(pd.read_csv(csv_fp))
    result_pd = pd.concat(pd_list).reset_index(drop=True)
    # result_pd['Pred_true'] = result_pd['Pred_PGr'] == result_pd['PGr']
    result_pd.to_csv(res_csv_stat_dir / 'result_all.csv', index=0)

    # stat for all samples
    exps = result_pd['Exp'].unique().tolist()
    ps = result_pd['P'].unique().tolist()
    pgrs = result_pd['PGr'].unique().tolist()
    markers = result_pd['Marker'].unique().tolist()
    ns = result_pd['N'].unique().tolist()
    exps.sort()
    ps.sort()
    pgrs.sort()
    markers.sort()
    ns.sort()

    # stat by marker
    for m in markers:
        m_pd = result_pd[result_pd['Marker'] == m].reset_index(drop=True)
        m_pd = m_pd.sort_values(['P'])
        m_pd.to_csv(res_csv_stat_dir / f'result_{m}.csv', index=0)

    # count
    ee = []
    mm = []
    pp = []
    ppgr = []
    nn = []
    counts = []
    areas = []
    pred_accs = []
    # for m in markers:
    #     m_pd = result_pd[result_pd['Marker'] == m]
    #     for p in ps:
    #         p_pd = m_pd[m_pd['P'] == p]
    for e in exps:
        e_pd = result_pd[result_pd['Exp'] == e]
        for n in ns:
            n_pd = e_pd[e_pd['N'] == n]
            count = n_pd.reset_index(drop=True).shape[0]
            area = n_pd['Area'].mean()
            # pred_acc = n_pd['Pred_true'].mean()
            if count != 0:
                ee.append(n_pd['Exp'].iloc[0])
                pp.append(n_pd['P'].iloc[0])
                ppgr.append(n_pd['PGr'].iloc[0])
                mm.append(n_pd['Marker'].iloc[0])
                nn.append(n_pd['N'].iloc[0])
                counts.append(count)
                areas.append(area)
                # pred_accs.append(pred_acc)
    count_pd = pd.DataFrame({
            'Exp': ee,
            'P': pp,
            'PGr': ppgr,
            'Marker': mm,
            'N': nn,
            'count': counts,
            'mean area': areas,
            # 'pred acc': pred_accs,
    })
    count_pd.to_csv(res_csv_stat_dir / 'result_all_count_area.csv', index=0)
    count_pd