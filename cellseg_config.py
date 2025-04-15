# Configuration file for the CellSeg project. Set experiment and dataset paths here.

### Dataset:

# exp = 'WJ-MSC-P57'
# dataset_dir = 'datasets/Cells_2.0_for_Ivan/masked_MSC/pics 2024-20240807T031703Z-002/pics 2024/WJ-MSC-P57'
# exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 1,
#                     '2024-05-03-wj-MSC-P57p5': 1,
#                     '2024-05-01-wj-MSC-P57p7': 2,
#                     '2024-05-04-wj-MSC-P57p9-sl2': 2,
#                     '2024-05-02-wj-MSC-P57p11': 3,
#                     '2024-05-03-wj-MSC-P57p13': 3,
#                     '2024-05-02-wj-MSC-P57p15sl2': 3}

exp = 'Microscopy_Ivan'
dataset_dir = 'datasets/Microscopy_Ivan'
exp_class_dict = {'25 wjMSC R1 (П57) p6 ctrl': 1,
                  '28 wjMSC R1 (П57) p6 H2O2 4h': 2,
                  '': 3}

### Model 

# # For single class
# model_name = 'DeepLabV3Plus_timm-efficientnet-b0_20241119_172741'
# model_dir = f'models/WJ-MSC-P57/{model_name}'

# For multiclass
model_name = 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611'
model_dir = f'models/WJ-MSC-P57/{model_name}'


model_results_dir = f'out/{exp}/{model_name}'
