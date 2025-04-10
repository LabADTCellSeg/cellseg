# Configuration file for the CellSeg project. Set experiment and dataset paths here.

# For single class
# exp = 'WJ-MSC-P57'
# model_name = 'DeepLabV3Plus_timm-efficientnet-b0_20241119_172741'
# dataset_dir = 'datasets/Cells_2.0_for_Ivan/masked_MSC/pics 2024-20240807T031703Z-002/pics 2024/WJ-MSC-P57'

# For multiclass
exp = 'WJ-MSC-P57'
model_name = 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611'
dataset_dir = 'datasets/Cells_2.0_for_Ivan/masked_MSC/pics 2024-20240807T031703Z-002/pics 2024/WJ-MSC-P57'

exp_dir = f'models/{exp}/{model_name}'
model_results_dir = f'out/{exp}/{model_name}'
