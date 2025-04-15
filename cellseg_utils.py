# This module contains common utility functions for the CellSeg project.

import re
import time
from datetime import datetime

import numpy as np


def get_str_timestamp(timestamp=None):
    """
    Returns a formatted timestamp string.
    Args:
        timestamp: Unix timestamp; if None, current time is used.
    Returns:
        String in the format "YYYYMMDD_HHMMSS".
    """
    if timestamp is None:
        timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
    return str_date_time


def get_all_fp_data(exps_dir, exp_class_dict):
    """
    Gathers all file pointer data from experiment directories.
    Args:
        exps_dir: Path to the experiments directory.
        exp_class_dict: Dictionary mapping experiment names to class labels.
    Returns:
        A list of dictionaries containing file pointers for each sample.
    """
    exps_dir_list = list()
    for v in exps_dir.iterdir():
        exps_dir_list.append(v.name)
    exps_dir_list.sort()

    all_fp_data = list()
    img_suffix = '.jpg'
    mask_suffix = '.png'
    for cur_exp in exps_dir_list:
        cur_exp_dir = exps_dir / cur_exp
        for mask_fp in list(cur_exp_dir.rglob('*p.jpg')):
            idx = mask_fp.name[:-len(mask_fp.suffix) - 1]

            m_fn = idx + 'm' + mask_suffix
            r_fn = idx + 'r' + img_suffix
            g_fn = idx + 'g' + img_suffix
            b_fn = idx + 'b' + img_suffix
            p_fn = idx + 'p' + img_suffix

            m_fp = cur_exp_dir / m_fn
            r_fp = cur_exp_dir / r_fn
            g_fp = cur_exp_dir / g_fn
            b_fp = cur_exp_dir / b_fn
            p_fp = cur_exp_dir / p_fn

            sample_data = dict(cls=exp_class_dict[cur_exp],
                               idx=idx,
                               mask_fp=m_fp,
                               r_fp=r_fp,
                               g_fp=g_fp,
                               b_fp=b_fp,
                               p_fp=p_fp)

            for k, v in sample_data.items():
                if '_fp' in k:
                    if not v.exists():
                        print(f'! not exists: {v}')

            all_fp_data.append(sample_data)
    return all_fp_data


def parse_filename_nd2(filename):
    """
    Parses an nd2 filename using a specific pattern.
    Args:
        filename: Name of the file.
    Returns:
        A tuple of extracted groups if the pattern matches, otherwise None.
    """
    pattern = r'^(.*?)_LF(\d+)-P(\d+)_(.*?)_(\d+)\.nd2$'
    match = re.match(pattern, filename)

    if match:
        group1 = match.group(1)
        group2 = int(match.group(3))
        group3 = match.group(4)
        group4 = int(match.group(5))

        return group1, group2, group3, group4
    else:
        return None


def get_classes_from_fps(fps, classes_groups=None):
    """
    Determines classes from file paths.
    Args:
        fps: List of file paths.
        classes_groups: Optional grouping of classes.
    Returns:
        A list of class labels.
    """
    classes = list()
    for fp in fps:
        fn = fp.split('/')[-1]
        exp, p, marker, n = parse_filename_nd2(fn)
        classes.append(p)

    if classes_groups is None:
        classes_groups = np.unique(classes)[..., np.newaxis]

    # max_class = len(classes_groups)
    for i in range(len(classes)):
        for classes_idx, classes_group in enumerate(classes_groups):
            if classes[i] in classes_group:
                classes[i] = classes_idx + 1
                # classes[i] = 1
                break

    return classes
