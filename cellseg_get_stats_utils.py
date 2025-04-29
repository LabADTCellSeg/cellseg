# This module provides utility functions to calculate cell statistics and draw ellipses based on the segmentation results.
from pathlib import Path
import multiprocessing as mp

import math
import numpy as np
import pandas as pd
import ast

import cv2
from PIL import Image

from scipy.spatial.distance import directed_hausdorff

import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil

from cellseg_utils import get_all_fp_data


def calculate_mse(contour_pts, center, axes, angle):
    # Generate the rotated rectangle that bounds the ellipse
    rect = (center, axes, angle)

    # Generate points on the boundary of the fitted ellipse
    generated_pts = cv2.boxPoints(rect).astype(np.int0)

    # Calculate mean squared error between contour points and ellipse boundary points
    mse = np.mean(np.sum((contour_pts - generated_pts) ** 2, axis=1))

    return mse


def get_cell_statistics(matrix, exp, exp_class_dir, p, pgr, marker, n,
                        passage_mask=None,
                        thresh_low=0.4,
                        thresh_margin=0.1,
                        thresh_tri=0.05):
    # Convert matrix to uint8 for OpenCV operations
    matrix_uint8 = (matrix * 255).astype(np.uint8)

    # Find external contours from the matrix
    contours, _ = cv2.findContours(
        matrix_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Lists to store statistics for each cell
    centers = []
    areas = []
    roundnesses = []
    ellipse_widths = []
    ellipse_heights = []
    angles = []
    hausdorff_distances = []

    # Lists for passage mask statistics and confidence metrics
    p1_list = []
    p2_list = []
    p3_list = []
    confidences = []
    margins = []
    pred_labels = []

    # Calculate statistics for each detected contour (cell)
    for contour in contours:
        # Calculate moments to determine centroid
        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            continue
        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])

        # Calculate area of the cell
        area = cv2.contourArea(contour)
        if area == 0:
            continue

        # Initialize raw probability measures
        p1 = p2 = p3 = 0.0

        # Calculate raw passage_mask overlaps if provided
        if passage_mask is not None:
            # Create binary mask of this contour as uint8
            contour_mask = np.zeros_like(passage_mask[0], dtype=np.uint8)
            cv2.drawContours(contour_mask, [contour], -1, color=1, thickness=cv2.FILLED)
            contour_mask_bool = contour_mask.astype(bool)

            # Intersection counts between contour and each class mask
            inter1 = np.logical_and(contour_mask_bool, passage_mask[0]).sum()
            inter2 = np.logical_and(contour_mask_bool, passage_mask[1]).sum()
            inter3 = np.logical_and(contour_mask_bool, passage_mask[2]).sum()

            # Normalize by contour area
            p1 = inter1 / area
            p2 = inter2 / area
            p3 = inter3 / area

            # Further normalize so that p1 + p2 + p3 = 1
            total_p = p1 + p2 + p3
            if total_p > 0:
                p1 /= total_p
                p2 /= total_p
                p3 /= total_p

        # Calculate roundness (shape compactness)
        perimeter = cv2.arcLength(contour, True)
        roundness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        # Fit an ellipse to the contour if enough points are available
        ellipse_width = ellipse_height = angle = np.nan
        hausdorff_distance = np.nan
        if contour.shape[0] > 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_width, ellipse_height, angle = ellipse[1][0], ellipse[1][1], ellipse[2]

            # Calculate Hausdorff distance between the fitted ellipse and the contour
            ellipse_center, ellipse_axes, ellipse_angle = ellipse
            ellipse_points = cv2.ellipse2Poly(
                (int(ellipse_center[0]), int(ellipse_center[1])),
                (int(ellipse_axes[0] / 2), int(ellipse_axes[1] / 2)),
                int(ellipse_angle), 0, 360, 10)
            hausdorff_distance = directed_hausdorff(
                ellipse_points.reshape(-1, 2), contour.reshape(-1, 2))[0]

        # Compile probability vector and compute confidence metrics
        p_vec = np.array([p1, p2, p3])
        max_i = np.argmax(p_vec)
        max_v = p_vec[max_i]
        sorted_v = np.sort(p_vec)
        second_v = sorted_v[-2]
        min_v = sorted_v[0]
        margin = max_v - second_v
        span = max_v - min_v

        # Determine prediction label based on thresholds
        if max_v < thresh_low or span < thresh_tri:
            pred_label = 'Unknown'
        elif margin < thresh_margin:
            # ambiguous top-2 classes
            sorted_desc = np.argsort(p_vec)[::-1]
            top2 = sorted_desc[:2]
            combo = sorted([int(top2[0]) + 1, int(top2[1]) + 1])
            pred_label = f"{combo[0]}/{combo[1]}"
            if pred_label == "1/3":
                pred_label = "3/1"
        else:
            pred_label = f"{max_i + 1}"

        # Append computed statistics
        centers.append(str((center_x, center_y)))
        areas.append(area)
        roundnesses.append(roundness)
        ellipse_widths.append(ellipse_width)
        ellipse_heights.append(ellipse_height)
        angles.append(angle)
        hausdorff_distances.append(hausdorff_distance)
        p1_list.append(p1)
        p2_list.append(p2)
        p3_list.append(p3)
        confidences.append(max_v)
        margins.append(margin)
        pred_labels.append(pred_label)

    # Build results dictionary
    res_dict = {
        'Exp': [exp] * len(centers),
        'Exp_dir': [exp_class_dir] * len(centers),
        'P': [p] * len(centers),
        'PGr': [pgr] * len(centers),
        'Marker': [marker] * len(centers),
        'N': [n] * len(centers),
        'Center': centers,
        'Area': areas,
        'Roundness': roundnesses,
        'Ellipse Width': ellipse_widths,
        'Ellipse Height': ellipse_heights,
        'Angle': angles,
        'Hausdorff Distance': hausdorff_distances,
        'PGr1_prob': p1_list,
        'PGr2_prob': p2_list,
        'PGr3_prob': p3_list,
        'Confidence': confidences,
        'Margin': margins,
        'Pred_PGr': pred_labels,
    }

    # Return DataFrame with all stats
    df = pd.DataFrame(res_dict)
    return df


def draw_ellipses(statistics_df, target_size=(1024, 1024), hd_max=10, thickness=1):
    # Create blank images for drawing ellipses and contours
    contours_image = np.zeros(target_size, dtype=np.uint8)
    ellipses_image = np.zeros(target_size, dtype=np.uint8)
    for _, row in statistics_df.iterrows():
        if row['Hausdorff Distance'] <= hd_max:
            center_x, center_y = ast.literal_eval(row['Center'])
            if not math.isnan(row['Ellipse Width']):
                ellipse_width = int(row['Ellipse Width'])
                ellipse_height = int(row['Ellipse Height'])
                angle = int(row['Angle'])
                # Draw ellipse
                cv2.ellipse(ellipses_image, (center_x, center_y),
                            (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, 1, -1)
                contours, _ = cv2.findContours(
                    ellipses_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contours_image, contours, -1, 1, thickness=thickness)
    return contours_image


def _process_sample(args):
    # Process a single sample: load image, mask and predicted masks, compute statistics and save outputs
    sample_data, pred_mask_fp_list, exp, exp_class_dict, channels, out_dir, er, dil, rewrite_existed, color_shift_list, color_shift_green, color_shift_white = args

    idx = sample_data['idx']
    image_fp = sample_data[f'{channels[0]}_fp']
    if 'mask_fp' in sample_data.keys():
        mask_fp = sample_data['mask_fp']
    else:
        mask_fp = Path(str(image_fp)[:-4 - len(channels[0])] + 'm.png')

    split_fp = str(image_fp).split('/')
    exp_class_dir = split_fp[-2]
    exp_class_value = exp_class_dict[exp_class_dir]

    exp_out_dir = out_dir / exp_class_dir

    exp_out_dir.mkdir(exist_ok=True, parents=True)
    if Path(mask_fp).exists():
        shutil.copy(mask_fp, exp_out_dir / mask_fp.name)
    else:
        img = np.asarray(Image.open(image_fp))
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        mask = Image.fromarray(mask)
        mask = mask.resize((img.shape[1], img.shape[0]))
        mask.save(exp_out_dir / mask_fp.name)

    for c_fp in [f'{c}_fp' for c in channels]:
        src_fp = sample_data.get(c_fp)
        if src_fp is not None:
            dst_fn = src_fp.name
            if Path(src_fp).exists():
                shutil.copy(src_fp, exp_out_dir / dst_fn)

    res_csv_fp = (exp_out_dir / f'{idx}result').with_suffix('.csv')
    res_img_fp = (exp_out_dir / f'{idx}result').with_suffix('.png')

    # Process and generate predicted masks if result files are missing or rewriting is allowed
    if not res_csv_fp.exists() or not res_img_fp.exists() or rewrite_existed:
        img = np.asarray(Image.open(image_fp))
        mask = np.asarray(Image.open(exp_out_dir / mask_fp.name))
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
        marker = channels[0]
        n = idx
        statistics_df = get_cell_statistics(
            main_mask, exp, exp_class_dir, p, pgr, marker, n, passage_mask=passage_mask,
            thresh_low=0.4,
            thresh_margin=0.1,
            thresh_tri=0.05)
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


def process_stat(exp, predicted_masks_dir, res_csv_stat_dir, out_dir, dataset_dir,
                 exp_class_dict=None, channels=None, ext='.jpg',
                 max_workers=8):
    dataset_dir = Path(dataset_dir)

    all_fp_data = get_all_fp_data(dataset_dir, exp_class_dict, channels=channels, ext=ext)
    all_fp_data = sorted(all_fp_data, key=lambda d: d['idx'])

    # Параметры цветовых сдвигов и морфологии
    color_shift_red = (+100, -100, -100)
    color_shift_green = (-200, +200, -200)
    color_shift_blue = (-100, -100, +100)
    color_shift_yellow = (+100, +100, -100)
    color_shift_white = (+255, +255, +255)
    color_shift_list = [color_shift_red, color_shift_blue, color_shift_yellow]
    er, dil = 20, 20
    num_samples = None
    rewrite_existed = True

    # Собираем аргументы для обработки
    samples_to_analyze = []
    for sample_data in all_fp_data:
        pred_mask_fp_list = [
            str(v) for v in (predicted_masks_dir / sample_data['exp_dir']).glob(f'{sample_data["idx"]}_*')
        ]
        pred_mask_fp_list.sort()
        if pred_mask_fp_list:
            samples_to_analyze.append([
                sample_data, pred_mask_fp_list,
                exp, exp_class_dict, channels,
                out_dir, er, dil, rewrite_existed,
                color_shift_list,
                color_shift_green,
                color_shift_white
            ])
    if num_samples:
        samples_to_analyze = samples_to_analyze[:num_samples]

    # Параллельная обработка с tqdm
    ctx = mp.get_context('forkserver')   # или 'spawn', если вы на Windows
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(_process_sample, args) for args in samples_to_analyze]
        for f in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing samples"):
            # Дождёмся завершения — если возникнет исключение, оно тут же вывалится
            f.result()

    # Аггрегация результатов
    csv_list = sorted([p for p in out_dir.rglob('*.csv') if 'csv_stat' not in str(p)])
    pd_list = [pd.read_csv(fp) for fp in csv_list]
    result_pd = pd.concat(pd_list, ignore_index=True)
    result_pd.to_csv(res_csv_stat_dir / 'result_all.csv', index=False)

    # По маркерам
    for m in sorted(result_pd['Marker'].unique()):
        dfm = result_pd[result_pd['Marker'] == m].sort_values('P').reset_index(drop=True)
        dfm.to_csv(res_csv_stat_dir / f'result_{m}.csv', index=False)

    # Подсчёты count & mean area
    records = []
    for e in sorted(result_pd['Exp'].unique()):
        edf = result_pd[result_pd['Exp'] == e]
        for n in sorted(edf['N'].unique()):
            sub = edf[edf['N'] == n]
            if not sub.empty:
                records.append({
                    'Exp': e,
                    'Exp_dir': sub['Exp_dir'].iat[0],
                    'P': sub['P'].iat[0],
                    'PGr': sub['PGr'].iat[0],
                    'Marker': sub['Marker'].iat[0],
                    'N': n,
                    'count': len(sub),
                    'mean area': sub['Area'].mean()
                })
    count_pd = pd.DataFrame(records)
    count_pd.to_csv(res_csv_stat_dir / 'result_all_count_area.csv', index=False)

    print('DONE!')
