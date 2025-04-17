# Configuration file for the CellSeg project. Set experiment and dataset paths here.

# Dataset:

# exp = 'WJ-MSC-P57'
# dataset_dir = 'datasets/Cells_2.0_for_Ivan/masked_MSC/pics 2024-20240807T031703Z-002/pics 2024/WJ-MSC-P57'
# exp_class_dict = {'2024-05-01-wj-MSC-P57p3': 1,
#                     '2024-05-03-wj-MSC-P57p5': 1,
#                     '2024-05-01-wj-MSC-P57p7': 2,
#                     '2024-05-04-wj-MSC-P57p9-sl2': 2,
#                     '2024-05-02-wj-MSC-P57p11': 3,
#                     '2024-05-03-wj-MSC-P57p13': 3,
#                     '2024-05-02-wj-MSC-P57p15sl2': 3}
# channels = ['b']
# extension = '.jpg'
# square_a, border = None, None
# # square_a, border = 1024, 0

exp = 'Microscopy_Ivan'
dataset_dir = 'datasets/Microscopy_Ivan'
exp_class_dict = {'25 wjMSC R1 (П57) p6 ctrl': 1,
                  '28 wjMSC R1 (П57) p6 H2O2 4h': 2}
classes = [1, 2, 3]
channels = ['b']
extension = '.jpg'
# square_a, border = None, None
square_a, border = 1024, 0

# exp = 'MSC_conf'
# dataset_dir = '/home/ivan/python/cellseg_export/datasets/2/MSC_conf'
# exp_class_dict = {
#     # 'o1_p2_c': 3,
#     'o1_p6': 3,
#     'o1_p10': 3,
#     'o2_p7': 3,
#     # 'o2_p7_c': 3,
#     'o2_p10': 3,
#     'o3_p2': 3,
#     # 'o3_p8_c': 3,
#     # 'y1_p2_c': 1,
#     'y1_p6': 1,
#     'y2_p2': 1,
#     'y2_p7': 1,
#     'y99_p3': 2,
#     'y99_p7': 2
# }
# classes = [1, 2, 3]
# channels = ['']
# extension = '.png'
# square_a, border = 1024, 0

# exp = 'MSC_light'
# dataset_dir = '/home/ivan/python/cellseg_export/datasets/2/MSC_light'
# exp_class_dict = {'o1_p4_HPL': 3,
#                   'o1_p8_FBS': 3,
#                   'Vas_p4': 3,
#                   'Vas_p7': 3,
#                   'Vas_p8': 3,
#                   'y1_p8': 1,
#                   'y1_p11': 1,
#                   'Y99p3': 2,
#                   'Y99p5': 2,
#                   'Y99p8sl2': 2}
# classes = [1, 2, 3]
# channels = ['']
# extension = '.jpg'
# # square_a, border = None, None
# square_a, border = 1024, 0

# Model

# For single class
model_name = 'DeepLabV3Plus_timm-efficientnet-b0_20241119_172741'  # CUDA
# model_name = 'DeepLabV3Plus_timm-efficientnet-b0_20241119_172741-cpu'  # CPU
model_dir = f'models/WJ-MSC-P57/{model_name}'

# # # For multiclass
# model_name = 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611'  # CUDA
# # model_name = 'MC_DeepLabV3Plus_timm-efficientnet-b0_20250116_130611-cpu'  # CPU
# model_dir = f'models/WJ-MSC-P57/{model_name}'

model_results_dir = f'out/{exp}/{model_name}'
