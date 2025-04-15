# This script runs a test experiment on the best saved model using the specified test configuration.

from cellseg_exp import test_exp
from cellseg_config import *


if __name__ == '__main__':
    use_all_images_for_test = True

    test_exp(model_dir, model_results_dir, dataset_dir, draw=True, use_all=use_all_images_for_test, batch_size=1, exp_class_dict=exp_class_dict)
