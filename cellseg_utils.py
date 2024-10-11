# import os
# import os.path as osp
import random
# import re
import time
from datetime import datetime

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import albumentations as A

from PIL import Image
import cv2
from tqdm import tqdm


def get_str_timestamp(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
    return str_date_time


def get_squares(orig_size, square_size, border):
    w, h = orig_size
    square_w, square_h = square_size

    # square_w_with_border = square_w + 2 * border
    # square_h_with_border = square_h + 2 * border
    # square_with_border_size = (square_w_with_border, square_h_with_border)

    square_w_num, square_h_num = int(np.ceil(w / square_w)), int(np.ceil(h / square_h))
    full_size = (square_w * square_w_num, square_h * square_h_num)
    full_size_with_borders = (full_size[0] + 2 * border, full_size[1] + 2 * border)

    squares = []
    for start_w_idx in range(square_w_num):
        start_w = start_w_idx * square_w
        end_w = (start_w_idx + 1) * square_w

        for start_h_idx in range(square_h_num):
            start_h = start_h_idx * square_h
            end_h = (start_h_idx + 1) * square_h

            square_coords = [start_w,
                             start_h,
                             end_w,
                             end_h]
            square_with_borders_coords = [start_w,
                                          start_h,
                                          end_w + 2 * border,
                                          end_h + 2 * border]
            square_with_borders_coords_rev = [start_w + border,
                                              start_h + border,
                                              end_w + border,
                                              end_h + border]

            square_info_dict = dict(w=start_w_idx,
                                    h=start_h_idx,
                                    square_coords=square_coords,
                                    square_with_borders_coords=square_with_borders_coords,
                                    square_with_borders_coords_rev=square_with_borders_coords_rev)
            squares.append(square_info_dict)

    return full_size, full_size_with_borders, squares


def split_on_squares(image, squares, param_name):
    img_sq_list = []
    for sq in squares:
        img_sq = image[:, sq[param_name][1]:sq[param_name][3], sq[param_name][0]:sq[param_name][2]].copy()
        img_sq_list.append(img_sq)

    return img_sq_list


def split_image(image, full_size, squares, border):
    c = image.shape[0]

    image_with_borders = []
    for c_idx in range(c):
        image_resized_cur_c = cv2.resize(image[c_idx], full_size, interpolation=cv2.INTER_NEAREST)
        image_with_borders_cur_c = cv2.copyMakeBorder(image_resized_cur_c, border, border, border, border,
                                                      cv2.BORDER_REFLECT, None)
        image_with_borders.append(image_with_borders_cur_c)
    image_with_borders = np.stack(image_with_borders)

    img_sq_list = split_on_squares(image_with_borders, squares, 'square_with_borders_coords')

    return image_with_borders, img_sq_list


def unsplit_image(img_sq_list, squares, param_name, border):
    w_num, h_num = squares[-1]['w'] + 1, squares[-1]['h'] + 1
    square_size = (squares[0]['square_coords'][2], squares[0]['square_coords'][3])
    result_size = (img_sq_list[0].shape[0], square_size[0] * h_num, square_size[1] * w_num)
    result = np.zeros(result_size)
    for sq, img_sq in zip(squares, img_sq_list):
        result[:, sq[param_name][1]:sq[param_name][3], sq[param_name][0]:sq[param_name][2]] = img_sq[:,
                                                                                              border:square_size[
                                                                                                         0] + border,
                                                                                              border:square_size[
                                                                                                         1] + border]
    return result


def draw_square(image, sq, color):
    for c_idx in range(len(color)):
        image[c_idx][sq[1]:sq[3], sq[0]] = color[c_idx]
        image[c_idx][sq[1], sq[0]:sq[2]] = color[c_idx]
        image[c_idx][sq[1]:sq[3], sq[2]] = color[c_idx]
        image[c_idx][sq[3], sq[0]:sq[2]] = color[c_idx]


def my_train_test_split(X, y, ratio_train, ratio_val, shuffle=True, seed=42):
    idx = np.arange(X.shape[0])
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(idx)

    limit_train = int(ratio_train * X.shape[0])
    limit_val = int((ratio_train + ratio_val) * X.shape[0])

    idx_train = idx[:limit_train]
    idx_val = idx[limit_train:limit_val]
    idx_test = idx[limit_val:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_all_fp_data(exps_dir, exp_class_dict):
    exps_dir_list = list()
    for v in exps_dir.iterdir():
        exps_dir_list.append(v.name)
    exps_dir_list.sort()

    all_fp_data = list()
    img_suffix = '.jpg'
    for cur_exp in exps_dir_list:
        cur_exp_dir = exps_dir / cur_exp
        for mask_fp in list(cur_exp_dir.rglob('*m.png')):
            idx = mask_fp.name[:-len(mask_fp.suffix) - 1]
            r_fn = idx + 'r' + img_suffix
            g_fn = idx + 'g' + img_suffix
            b_fn = idx + 'b' + img_suffix
            p_fn = idx + 'p' + img_suffix

            r_fp = cur_exp_dir / r_fn
            g_fp = cur_exp_dir / g_fn
            b_fp = cur_exp_dir / b_fn
            p_fp = cur_exp_dir / p_fn

            sample_data = dict(cls=exp_class_dict[cur_exp],
                               mask_fp=mask_fp,
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


class CellDataset4(BaseDataset):
    def __init__(
            self,
            all_fp_data,
            exp_class_dict,
            full_size,
            add_shadow_to_img=False,
            squares=None,
            border=None,
            channels=3,
            classes_num=2,
            classes=None,
            augmentation=None,
            preprocessing=None,
            target_size=None
    ):
        self.all_fp_data = all_fp_data
        self.exp_class_dict = exp_class_dict
        self.full_size = full_size

        self.add_shadow_to_img = add_shadow_to_img

        self.squares = squares
        self.border = border
        self.channels = channels

        self.classes_num = classes_num
        self.classes = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.target_size = target_size

        self.contour_thickness = 2

        if self.border is None:
            self.border = 0

        self.images = list()
        self.masks = list()
        self.shadows = list()
        self.squares_info = list()
        for fp_data in tqdm(self.all_fp_data):
            img = self.read_image(fp_data).astype(np.uint8)
            mask = self.read_mask(fp_data) / 255
            assert img.shape[:-1] == mask.shape[:-1]

            mask = self._prepare_mask(mask).astype(np.uint8)
            shadow = self.read_shadow(fp_data).astype(np.uint8) if self.add_shadow_to_img else None

            self.images.append(img)
            self.masks.append(mask)
            self.shadows.append(shadow)
            self.squares_info.append(None)

    def read_image(self, fp_data, channels=None, convert=None):
        if channels is None:
            channels = ['r', 'g', 'b']

        c_stack = list()
        for c_idx, c in enumerate(channels):
            c_img = Image.open(fp_data[f'{c}_fp'])

            new_width = c_img.size[0] // 32 * 32
            new_height = c_img.size[0] // 32 * 32
            c_img = c_img.resize((new_width, new_height))

            if self.target_size:
                c_img = c_img.resize(self.target_size)

            if convert:
                c_img = np.asarray(c_img.convert(convert))
            else:
                c_img = np.asarray(c_img)[..., c_idx]
            c_stack.append(c_img)

        img_np = np.stack(c_stack, axis=-1)

        return img_np

    def read_mask(self, fp_data):
        return self.read_image(fp_data, channels=['mask'])

    def read_shadow(self, fp_data):
        return self.read_image(fp_data, channels=['p'], convert='L')

    def _prepare_mask(self, mask):
        mask_contour = mask.copy()
        mask_contour = mask_contour.astype('uint8')
        contours, hierarchy = cv2.findContours(mask_contour,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, -1, 2, self.contour_thickness)
        # mask_contour = mask_contour.transpose(2, 0, 1)
        # mask_contour = mask_contour
        mask = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=mask.dtype)
        mask[..., 0:1][mask_contour == 1] = 1
        mask[..., 1:2][mask_contour == 2] = 1

        return mask

    def aug(self, img, mask, shadow=False):
        if self.augmentation or self.preprocessing:
            img_orig_dtype = img.dtype
            mask_orig_dtype = mask.dtype

            if shadow is None:
                if self.augmentation:
                    sample = self.augmentation(image=img.astype(np.uint8), mask=mask.astype(np.uint8))
                    img, mask = sample['image'], sample['mask']

                if self.preprocessing:
                    sample = self.preprocessing(image=img.astype(np.uint8), mask=mask.astype(np.uint8))
                    img, mask = sample['image'], sample['mask']

                img = img.astype(img_orig_dtype)
                mask = mask.astype(mask_orig_dtype)

            else:
                if self.augmentation:
                    sample = self.augmentation(image=img.astype(np.uint8), mask=mask.astype(np.uint8),
                                               shadow=shadow.astype(np.uint8))
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']

                if self.preprocessing:
                    sample = self.preprocessing(image=img.astype(np.uint8), mask=mask.astype(np.uint8),
                                                shadow=shadow.astype(np.uint8))
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']

                # shadow = np.asarray(Image.fromarray(np.dstack[np.uint8(shadow)]*3)).convert('L'))
                shadow = shadow[..., 0].astype(img_orig_dtype)
                img = img.astype(img_orig_dtype)
                img = np.dstack([img, shadow])
                mask = mask.astype(mask_orig_dtype)

        return img, mask

    # def _create_squares(self):
    #     images = list()
    #     masks = list()
    #     squares_info = list()
    #     for fp_data in tqdm(self.all_fp_data):
    #         img = self.read_image(fp_data)
    #         mask = self.read_mask(fp_data)
    #         assert img.shape[:-1] == mask.shape[:-1]

    #         mask = self._prepare_mask(mask / 255)

    #         # split
    #         _, img_sq_list = split_image(img, self.full_size,
    #                                      self.squares, self.border)
    #         _, mask_sq_list = split_image(mask, self.full_size,
    #                                       self.squares, self.border)
    #         if self.add_shadow_to_img:
    #             shadow = self.read_shadow(fp_data)
    #             _, shadow_sq_list = split_image(shadow, self.full_size,
    #                           self.squares, self.border)
    #         else:
    #             shadow_sq_list = [None] * len(img_sq_list)

    #         for img_sq, msk_sq, shd_sq, sq in zip(img_sq_list,
    #                                       mask_sq_list, shadow_sq_list,
    #                                       self.squares):

    #             img_sq, msk_sq = self.aug(img, mask, shadow=shd_sq)
    #             img_sq = img_sq.transpose(2, 0, 1)
    #             msk_sq = msk_sq.transpose(2, 0, 1)

    #             images.append(img_sq.astype(np.uint8))
    #             masks.append(msk_sq.astype(np.uint8))
    #             # cur_cq = sq.copy()
    #             # cur_cq['fp'] = fp
    #             squares_info.append(sq)

    #     # images = np.stack(images, axis=0)
    #     # masks = np.stack(masks, axis=0)
    #     return images, masks, squares_info

    def __getitem__(self, i):
        img, mask = self.aug(self.images[i], self.masks[i], shadow=self.shadows[i])

        return img.transpose(2, 0, 1).astype(np.float32), mask.transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.images)


def get_training_augmentation(target_size=None):
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=0, p=1, border_mode=0),

        # A.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        A.RandomResizedCrop(*target_size, scale=(0.6, 1.0), p=0.5),

        A.GaussNoise(p=0.2),
        # A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.HueSaturationValue(p=0.9),
    ]
    if target_size is not None:
        train_transform.append(A.Resize(*target_size, always_apply=True))
    return A.Compose(train_transform, is_check_shapes=False, additional_targets={'shadow': 'image'})


def get_validation_augmentation(target_size=None):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(384, 480, always_apply=True, border_mode=0),
        # A.Resize(384, 480)
    ]
    if target_size is not None:
        test_transform.append(A.Resize(*target_size, always_apply=True))
    return A.Compose(test_transform, is_check_shapes=False, additional_targets={'shadow': 'image'})


def to_tensor(x, **kwargs):
    return x.astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: Amentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform, is_check_shapes=False)


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum()

    loss = (1 - ((2. * intersection + smooth) / (
            pred.sum() + target.sum() + smooth)))

    return loss.mean()


class BCEDiceLoss:
    __name__ = 'bce_dice'

    def __init__(self, bce_weight=0.5):
        self.bce_weight = bce_weight
        self.device = 'cpu'

    def __call__(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='mean')

        pred = torch.sigmoid(pred)
        dice = dice_loss(pred, target)

        loss = bce * self.bce_weight + dice * (1 - self.bce_weight)

        return loss.to(self.device)

    def to(self, device):
        self.device = device


# def parse_filename(filename):
#     # Определяем шаблон для поиска
#     pattern = r'^(.*?)_LF(\d+)-P(\d+)_(.*?)_(\d+)\.nd2$'

#     # Применяем регулярное выражение к имени файла
#     match = re.match(pattern, filename)

#     if match:
#         # Получаем группы из регулярного выражения
#         group1 = match.group(1)
#         group2 = int(match.group(3))
#         group3 = match.group(4)
#         group4 = int(match.group(5))

#         return group1, group2, group3, group4
#     else:
#         return None


def get_classes_from_fps(fps, classes_groups=None):
    classes = list()
    for fp in fps:
        fn = fp.split('/')[-1]
        exp, p, marker, n = parse_filename(fn)
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


def prepare_data(p, images_num=None, shuffle=True):
    all_fp_data = get_all_fp_data(p.dataset_dir, p.exp_class_dict)
    all_fp_data = all_fp_data[:images_num]
    if shuffle:
        random.shuffle(all_fp_data)

    total_len = len(all_fp_data)
    train_num = int(total_len * p.ratio_train)
    val_num = int(total_len * p.ratio_val)
    test_num = total_len - val_num

    train_fp_data = all_fp_data[:train_num]
    val_fp_data = all_fp_data[train_num:train_num + val_num]
    test_fp_data = all_fp_data[train_num + val_num:]

    mask_img = Image.open(all_fp_data[0]['mask_fp'])
    w, h = mask_img.size[0], mask_img.size[1]

    k = 8

    w, h = int(w / k // 32 * 32), int(h / k // 32 * 32)
    target_size = (w, h)

    if p.square_a is None:
        full_size, squares = None, None
    else:
        square_w, square_h = p.square_a, p.square_a
        square_size = (square_w, square_h)
        full_size, full_size_with_borders, squares = get_squares(target_size, square_size, p.border)

    add_shadow_to_img = True

    preprocessing_fn = smp.encoders.get_preprocessing_fn(p.ENCODER, p.ENCODER_WEIGHTS)
    # preprocessing_fn = None
    preprocessing = get_preprocessing(preprocessing_fn)

    train_dataset = CellDataset4(train_fp_data,
                                 p.exp_class_dict,
                                 full_size=full_size,
                                 add_shadow_to_img=add_shadow_to_img,
                                 squares=squares,
                                 border=p.border,
                                 channels=None,
                                 classes_num=2,
                                 augmentation=get_training_augmentation(target_size=target_size),
                                 preprocessing=preprocessing,
                                 classes=None,
                                 target_size=target_size
                                 )

    valid_dataset = CellDataset4(val_fp_data,
                                 p.exp_class_dict,
                                 full_size=full_size,
                                 add_shadow_to_img=add_shadow_to_img,
                                 squares=squares,
                                 border=p.border,
                                 channels=None,
                                 classes_num=2,
                                 augmentation=get_validation_augmentation(target_size=target_size),
                                 preprocessing=preprocessing,
                                 classes=None,
                                 target_size=target_size
                                 )

    test_dataset = CellDataset4(test_fp_data,
                                p.exp_class_dict,
                                full_size=full_size,
                                add_shadow_to_img=add_shadow_to_img,
                                squares=squares,
                                border=p.border,
                                channels=None,
                                classes_num=2,
                                augmentation=get_validation_augmentation(target_size=target_size),
                                preprocessing=preprocessing,
                                classes=None,
                                target_size=target_size
                                )

    return train_dataset, valid_dataset, test_dataset
