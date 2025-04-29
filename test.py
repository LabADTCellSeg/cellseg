from cm import create_cms
from cellseg_get_stats_utils import process_stat
from cellseg_exp import test_exp
from cellseg_config import get_config
from pathlib import Path
import multiprocessing as mp
mp.set_start_method('fork', force=True)


if __name__ == '__main__':

    dataset_cfg_name_list = [
        # 'WJ-MSC-P57',
        'WJ-MSC-P57 512',
        # 'Microscopy_Ivan',
        # 'MSC_light',
        # 'MSC_conf'
    ]

    # old dataset format
    model_name_list = [
        # 'DeepLabV3Plus_timm-efficientnet-b0_20241119_172741', 
        # 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250420_161017',
        # 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250420_162747',
    ]

    # new dataset format
    model_name_list = [
        # 'MC_DeepLabV3Plus_timm-efficientnet-b4_20250422_134708',
        'MC_DeepLabV3Plus_timm-efficientnet-b4_20250422_134939'
    ]

    use_all = False
    max_workers = 4  # Maximum number of parallel workers (typically equal to CPU cores)

    true_classes = [1, 2, 3]
    pred_classes = ["1", "1/2", "2", "2/3", "3", "3/1", "Unknown"]

    create_mask_images = True
    create_stat_tables = True
    create_cm_images = True

    for dataset_cfg_name in dataset_cfg_name_list:
        for model_name in model_name_list:
            dataset_cfg, model_cfg = get_config(dataset_cfg_name, model_name)
            if create_mask_images:
                test_exp(
                    model_dir=model_cfg['model_dir'],
                    out_dir=model_cfg['model_results_dir'],
                    dataset_dir=dataset_cfg['dataset_dir'],
                    classes=dataset_cfg['classes'],
                    draw=True,
                    use_all=use_all,
                    batch_size=1,
                    exp_class_dict=dataset_cfg['exp_class_dict'],
                    channels=dataset_cfg['channels'],
                    ext=dataset_cfg['extension'],
                    square_a=dataset_cfg['square_a'],
                    border=dataset_cfg['border'],
                )

            model_cfg['model_results_dir'] = Path(model_cfg['model_results_dir'])
            stats_out_dir = Path(model_cfg['model_results_dir']) / 'stats_results'
            predicted_masks_dir = model_cfg['model_results_dir'] / 'predicted_masks'
            res_csv_stat_dir = stats_out_dir / 'csv_stat'
            res_csv_stat_dir.mkdir(exist_ok=True, parents=True)
            if create_stat_tables:
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

            csv_fp = res_csv_stat_dir / 'result_all.csv'
            cm_out_dir = Path(model_cfg['model_results_dir']) / 'cm'
            if create_cm_images:
                create_cms(csv_fp=csv_fp,
                           out_dir=cm_out_dir,
                           exp_class_dict=None,
                           true_classes=true_classes,
                           pred_classes=pred_classes)
