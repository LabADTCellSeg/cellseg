# This script calculates cell statistics from segmentation results and saves the results as CSV and image files.
from pathlib import Path

from cellseg_config import get_config
from cellseg_get_stats_utils import process_stat

if __name__ == '__main__':

    dataset_cfg_name_list = [
        'WJ-MSC-P57',
        # 'Microscopy_Ivan',
        # 'MSC_light',
        # 'MSC_conf'
    ]

    model_name_list = [
        # 'DeepLabV3Plus_timm-efficientnet-b0_20241119_172741',
        'MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250420_161017',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250420_162322',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250420_162747',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b4_20250422_134708',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b4_20250422_134939'
    ]

    max_workers = 8  # Maximum number of parallel workers (typically equal to CPU cores)

    for dataset_cfg_name in dataset_cfg_name_list:
        for model_name in model_name_list:
            dataset_cfg, model_cfg = get_config(dataset_cfg_name, model_name)

            model_cfg['model_results_dir'] = Path(model_cfg['model_results_dir'])
            out_dir = Path(model_cfg['model_results_dir']) / 'stats_results'
            predicted_masks_dir = model_cfg['model_results_dir'] / 'predicted_masks'

            res_csv_stat_dir = out_dir / 'csv_stat'
            res_csv_stat_dir.mkdir(exist_ok=True, parents=True)
            process_stat(exp=dataset_cfg_name,
                         predicted_masks_dir=predicted_masks_dir,
                         res_csv_stat_dir=res_csv_stat_dir,
                         out_dir=model_cfg['model_results_dir'],
                         dataset_dir=dataset_cfg['dataset_dir'],
                         exp_class_dict=dataset_cfg['exp_class_dict'],
                         channels=dataset_cfg['channels'],
                         ext=dataset_cfg['extension'],
                         max_workers=max_workers
                         )
