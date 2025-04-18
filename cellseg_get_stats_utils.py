# This module provides utility functions to calculate cell statistics and draw ellipses based on the segmentation results.

import math
import numpy as np
import pandas as pd
import ast

import cv2

from scipy.spatial.distance import directed_hausdorff


def calculate_mse(contour_pts, center, axes, angle):
    # Generate the rotated rectangle that bounds the ellipse
    rect = (center, axes, angle)

    # Generate points on the boundary of the fitted ellipse
    generated_pts = cv2.boxPoints(rect).astype(np.int0)

    # Calculate mean squared error between contour points and ellipse boundary points
    mse = np.mean(np.sum((contour_pts - generated_pts) ** 2, axis=1))

    return mse


def calc_ps(passage_mask, contour):
    # Initialize a contour mask for the passage mask
    passage_mask_contour = np.zeros(
        (passage_mask.shape[1], passage_mask.shape[2]))
    cv2.drawContours(passage_mask_contour, [
                     contour], -1, color=1, thickness=cv2.FILLED)
    p1 = passage_mask[0][np.logical_and(
        passage_mask_contour == 1, passage_mask[0] == 1)].sum()
    p2 = passage_mask[1][np.logical_and(
        passage_mask_contour == 1, passage_mask[1] == 1)].sum()
    p3 = passage_mask[2][np.logical_and(
        passage_mask_contour == 1, passage_mask[2] == 1)].sum()
    # total = matrix[np.logical_and(passage_mask_contour == 1, matrix == 1)].sum()
    total = p1 + p2 + p3

    p1 /= total
    p2 /= total
    p3 /= total

    return p1, p2, p3


def get_cell_statistics(matrix, exp, exp_class_dir, p, pgr, marker, n, passage_mask=None):
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

    if passage_mask is not None:
        p1_list = []
        p2_list = []
        p3_list = []

    # Calculate statistics for each detected contour (cell)
    for contour in contours:
        # Calculate moments to determine centroid
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            # Calculate area of the cell
            area = cv2.contourArea(contour)
            if passage_mask is not None:
                p1, p2, p3 = calc_ps(passage_mask, contour)

            # Calculate roundness (a measure of shape compactness)
            perimeter = cv2.arcLength(contour, True)
            roundness = (4 * np.pi * area) / (perimeter ** 2)

            # Fit an ellipse to the contour if enough points are available
            if contour.shape[0] > 5:
                ellipse = cv2.fitEllipse(contour)

                ellipse_width = ellipse[1][0]
                ellipse_height = ellipse[1][1]

                # Extract the angle of the fitted ellipse
                angle = ellipse[2]

                # Calculate Hausdorff distance between the fitted ellipse and the contour
                ellipse_center, ellipse_axes, ellipse_angle = ellipse
                ellipse_points = cv2.ellipse2Poly((int(ellipse_center[0]), int(ellipse_center[1])),
                                                  (int(ellipse_axes[0] / 2), int(ellipse_axes[1] / 2)),
                                                  int(ellipse_angle), 0, 360, 10)
                hausdorff_distance = directed_hausdorff(
                    ellipse_points.reshape(-1, 2), contour.reshape(-1, 2))[0]

                centers.append(str((center_x, center_y)))
                areas.append(area)
                roundnesses.append(roundness)
                ellipse_widths.append(ellipse_width)
                ellipse_heights.append(ellipse_height)
                angles.append(angle)
                hausdorff_distances.append(hausdorff_distance)
                if passage_mask is not None:
                    p1_list.append(p1)
                    p2_list.append(p2)
                    p3_list.append(p3)

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
    }
    if passage_mask is not None:
        res_dict['PGr1_prob'] = p1_list
        res_dict['PGr2_prob'] = p2_list
        res_dict['PGr3_prob'] = p3_list
        pred_p_all = np.stack([p1_list, p2_list, p3_list], axis=0)
        pred_p = np.argmax(pred_p_all, axis=0) + 1
        res_dict['Pred_PGr'] = pred_p.tolist()

    # Create and return a DataFrame with the computed statistics
    df = pd.DataFrame(res_dict)

    return df


def draw_ellipses(statistics_df, target_size=(1024, 1024), hd_max=10, thickness=1):
    # Create blank images for drawing ellipses and contours
    contours_image = np.zeros(target_size, dtype=np.uint8)
    ellipses_image = np.zeros(target_size, dtype=np.uint8)
    # Draw ellipses on the image for rows with low Hausdorff distance (good fit)
    for _, row in statistics_df.iterrows():
        if row['Hausdorff Distance'] <= hd_max:
            # Extract ellipse parameters from the DataFrame
            center_x, center_y = ast.literal_eval(row['Center'])
            if not math.isnan(row['Ellipse Width']):
                ellipse_width = int(row['Ellipse Width'])
                ellipse_height = int(row['Ellipse Height'])
                angle = int(row['Angle'])

                # Draw the ellipse on the blank image
                cv2.ellipse(ellipses_image, (center_x, center_y),
                            (ellipse_width // 2, ellipse_height // 2), angle, 0, 360, 1, -1)

                contours, hierarchy = cv2.findContours(
                    ellipses_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                _ = cv2.drawContours(contours_image, contours, -1, 1, thickness=thickness)

    return contours_image
