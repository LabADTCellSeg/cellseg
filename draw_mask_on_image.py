# !pip install nd2
# !pip install numpy
# !pip install matplotlib

import os
import os.path as osp

import nd2
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    show = False
    save = True

    images_dir = 'datasets/MSC/30_04_2023-LF1-P6-P21'
    masks_dir = 'datasets/MSC/30_04_2023-LF1-P6-P21_masks'
    out_dir = 'datasets/MSC/30_04_2023-LF1-P6-P21_orig_with_masks'
    os.makedirs(out_dir, exist_ok=True)

    nd2_max = 4095
    png_max = 255
    image_channel = 0  # 0 or 1
    color_shift = (-100, +100, -100)  # (R, G, B)

    fn_list = [v for v in os.listdir(images_dir) if v.endswith('.nd2')]
    fn_list.sort()

    print('image processing:')
    if show:
        plt.subplot(111)
    for idx, fn in enumerate(fn_list):
        print(f'\r{" " * 80}', end='', flush=True)
        print(f'\r{idx + 1:4d}/{len(fn_list):4d}: {fn}', end='', flush=True)
        fp = osp.join(images_dir, fn)
        mask_fp = osp.join(masks_dir, f'{fn}mask.png')

        image = nd2.imread(fp).astype(np.float32)
        mask = plt.imread(mask_fp)[..., 0]

        img1 = image[image_channel].copy() / nd2_max * png_max
        img3 = np.stack([img1] * 3, axis=2)

        mask_indices = mask == 1
        for c_idx, c in enumerate(color_shift):
            img3[..., c_idx][mask_indices] += c

        img3 = img3.clip(0, png_max).astype(np.uint8)

        if show:
            plt.clf()
            plt.title(fn)
            plt.imshow(img3)
            plt.show(block=False)
            plt.pause(0.001)
        if save:
            plt.imsave(osp.join(out_dir, f'{fn}.png'), img3)

    print('\nDONE')
