# %% imports and main parameters
import os
import os.path as osp
import torch

from cellseg_exp import test_exp
from cellseg_utils import get_str_timestamp

CASCADE = False

if CASCADE:
    server_name = 'CASCADE'
    TORCH_HUB_DIR = '/storage0/pia/python/hub/'
    torch.hub.set_dir(TORCH_HUB_DIR)
    root_dir = '/storage0/pia/python/cellseg/'
else:
    server_name = 'seth'
    root_dir = '.'

exp_dir = 'out_CASCADE/out/WJ-MSC-P57/DeepLabV3Plus_timm-efficientnet-b0_20241119_172741'
out_dir = osp.join(exp_dir, 'test_results')
dataset_dir = 'datasets/Cells_2.0_for_Ivan/masked_MSC/pics 2024-20240807T031703Z-002/pics 2024/WJ-MSC-P57'
test_exp(exp_dir, out_dir, dataset_dir, draw=True)