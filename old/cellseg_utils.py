import os
import os.path as osp
import random
import re
import time
from datetime import datetime

import nd2
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu

import cv2
from tqdm import tqdm


def get_str_timestamp(timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    date_time = datetime.fromtimestamp(timestamp)
    str_date_time = date_time.strftime("%Y%m%d_%H%M%S")
    return str_date_time


def get_squares(orig_size, square_size, border, ):
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
    w_num, h_num = squares[-1]['w'] + 1, squares[-1]['w'] + 1
    square_size = (squares[0]['square_coords'][2], squares[0]['square_coords'][3])
    result_size = (img_sq_list[0].shape[0], square_size[0] * w_num, square_size[1] * h_num)
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


class CellDatasetMC(BaseDataset):
    def __init__(
            self,
            images_fps,
            masks_fps,
            squares,
            border,
            channels,
            classes_num,
            full_size,
            augmentation=None,
            preprocessing=None,
            classes=None
    ):
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        self.squares = squares
        self.border = border
        self.channels = channels
        self.classes_num = classes_num
        self.full_size = full_size

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.classes = classes

        # self.images, self.masks, self.squares_info = self._create_squares()
        self.images, self.masks, self.squares_info = self._create_squares_multiclass()

    def _create_squares(self):
        images = list()
        masks = list()
        squares_info = list()
        for fp, mask_fp in tqdm(zip(self.images_fps, self.masks_fps)):
            img = nd2.imread(fp)
            img = img / 4095

            assert img.shape == (self.channels, 1024, 1024)

            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
            mask = mask / 255

            mask_contour = mask.copy()
            mask_contour = mask_contour.transpose(1, 2, 0).astype('uint8')
            contours, hierarchy = cv2.findContours(mask_contour,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(mask_contour, contours, -1, 2, 2)

            # mask = mask_contour.transpose(2, 0, 1).astype('float64')
            # assert mask.shape == (1, 1024, 1024)

            mask_contour = mask_contour.transpose(2, 0, 1).astype('float64')
            mask = np.zeros((2, mask.shape[1], mask.shape[2]))
            mask[0:1][mask_contour == 1] = 1
            mask[1:2][mask_contour == 2] = 1
            assert mask.shape == (self.classes_num, 1024, 1024)

            # split
            _, img_sq_list = split_image(img, self.full_size,
                                         self.squares, self.border)
            _, mask_sq_list = split_image(mask, self.full_size,
                                          self.squares, self.border)
            for img_sq, msk_sq, sq in zip(img_sq_list,
                                          mask_sq_list,
                                          self.squares):
                if self.augmentation:
                    sample = self.augmentation(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                if self.preprocessing:
                    sample = self.preprocessing(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                images.append(img_sq.astype(np.float32))
                masks.append(msk_sq.astype(np.float32))
                cur_cq = sq.copy()
                cur_cq['fp'] = fp
                squares_info.append(cur_cq)

        return images, masks, squares_info

    def _create_squares_multiclass(self):
        images = list()
        masks = list()
        squares_info = list()

        if self.classes is None:
            self.classes = list()
            for _ in range(len(self.images_fps)):
                self.classes.append(1)

        for fp, mask_fp, cl in tqdm(zip(self.images_fps, self.masks_fps, self.classes)):
            img = nd2.imread(fp)
            img = img / 4095

            assert img.shape == (self.channels, 1024, 1024)

            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
            mask = mask / 255

            mask_contour = mask.copy()
            mask_contour = mask_contour.transpose(1, 2, 0).astype('uint8')
            contours, hierarchy = cv2.findContours(mask_contour,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(mask_contour, contours, -1, 2, 2)

            # mask = mask_contour.transpose(2, 0, 1).astype('float64')
            # assert mask.shape == (1, 1024, 1024)

            mask_contour = mask_contour.transpose(2, 0, 1).astype('float64')
            mask = np.zeros((self.classes_num, mask.shape[1], mask.shape[2]))
            if cl == 1:
                cl = 0
            mask[1:2][mask_contour == 2] = 1
            mask[cl:cl + 1][mask_contour == 1] = 1
            assert mask.shape == (self.classes_num, 1024, 1024)

            # split
            _, img_sq_list = split_image(img, self.full_size,
                                         self.squares, self.border)
            _, mask_sq_list = split_image(mask, self.full_size,
                                          self.squares, self.border)
            for img_sq, msk_sq, sq in zip(img_sq_list,
                                          mask_sq_list,
                                          self.squares):
                if self.augmentation:
                    sample = self.augmentation(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                if self.preprocessing:
                    sample = self.preprocessing(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                images.append(img_sq.astype(np.float32))
                masks.append(msk_sq.astype(np.float32))
                cur_cq = sq.copy()
                cur_cq['fp'] = fp
                squares_info.append(cur_cq)

        return images, masks, squares_info

    def __getitem__(self, i):

        return self.images[i], self.masks[i]

    def __len__(self):
        return len(self.images)


class CellDatasetMCnpy(BaseDataset):
    def __init__(
            self,
            images_fps,
            masks_fps,
            squares,
            border,
            channels,
            classes_num,
            full_size,
            augmentation=None,
            preprocessing=None,
            classes=None
    ):
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        self.squares = squares
        self.border = border
        self.channels = channels
        self.classes_num = classes_num
        self.full_size = full_size

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.classes = classes

        # self.images, self.masks, self.squares_info = self._create_squares()
        self.images, self.masks, self.squares_info = self._create_squares_multiclass()

    def _create_squares(self):
        images = list()
        masks = list()
        squares_info = list()
        for fp, mask_fp in tqdm(zip(self.images_fps, self.masks_fps)):
            img = nd2.imread(fp)
            img = img / 4095

            assert img.shape == (self.channels, 1024, 1024)

            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
            mask = mask / 255

            mask_contour = mask.copy()
            mask_contour = mask_contour.transpose(1, 2, 0).astype('uint8')
            contours, hierarchy = cv2.findContours(mask_contour,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(mask_contour, contours, -1, 2, 2)

            # mask = mask_contour.transpose(2, 0, 1).astype('float64')
            # assert mask.shape == (1, 1024, 1024)

            mask_contour = mask_contour.transpose(2, 0, 1).astype('float64')
            mask = np.zeros((2, mask.shape[1], mask.shape[2]))
            mask[0:1][mask_contour == 1] = 1
            mask[1:2][mask_contour == 2] = 1
            assert mask.shape == (self.classes_num, 1024, 1024)

            # split
            _, img_sq_list = split_image(img, self.full_size,
                                         self.squares, self.border)
            _, mask_sq_list = split_image(mask, self.full_size,
                                          self.squares, self.border)
            for img_sq, msk_sq, sq in zip(img_sq_list,
                                          mask_sq_list,
                                          self.squares):
                if self.augmentation:
                    sample = self.augmentation(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                if self.preprocessing:
                    sample = self.preprocessing(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                images.append(img_sq.astype(np.float32))
                masks.append(msk_sq.astype(np.float32))
                cur_cq = sq.copy()
                cur_cq['fp'] = fp
                squares_info.append(cur_cq)

        return images, masks, squares_info

    def _create_squares_multiclass(self):
        images = list()
        masks = list()
        squares_info = list()

        if self.classes is None:
            self.classes = list()
            for _ in range(len(self.images_fps)):
                self.classes.append(1)

        for fp, mask_fp, cl in tqdm(zip(self.images_fps, self.masks_fps, self.classes)):
            img = nd2.imread(fp)
            img = img / 4095

            with open(fp+'.npy', 'rb') as f:
                img_2 = np.load(f)
                img = np.vstack([img, img_2])

            assert img.shape == (self.channels, 1024, 1024)

            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
            mask = mask / 255

            mask_contour = mask.copy()
            mask_contour = mask_contour.transpose(1, 2, 0).astype('uint8')
            contours, hierarchy = cv2.findContours(mask_contour,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(mask_contour, contours, -1, 2, 2)

            # mask = mask_contour.transpose(2, 0, 1).astype('float64')
            # assert mask.shape == (1, 1024, 1024)

            mask_contour = mask_contour.transpose(2, 0, 1).astype('float64')
            mask = np.zeros((self.classes_num, mask.shape[1], mask.shape[2]))
            if cl == 1:
                cl = 0
            mask[1:2][mask_contour == 2] = 1
            mask[cl:cl + 1][mask_contour == 1] = 1
            assert mask.shape == (self.classes_num, 1024, 1024)

            # split
            _, img_sq_list = split_image(img, self.full_size,
                                         self.squares, self.border)
            _, mask_sq_list = split_image(mask, self.full_size,
                                          self.squares, self.border)
            for img_sq, msk_sq, sq in zip(img_sq_list,
                                          mask_sq_list,
                                          self.squares):
                if self.augmentation:
                    sample = self.augmentation(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                if self.preprocessing:
                    sample = self.preprocessing(image=img_sq.astype('float32'), mask=msk_sq.astype('float32'))
                    img_sq, msk_sq = sample['image'], sample['mask']

                images.append(img_sq.astype(np.float32))
                masks.append(msk_sq.astype(np.float32))
                cur_cq = sq.copy()
                cur_cq['fp'] = fp
                squares_info.append(cur_cq)

        return images, masks, squares_info

    def __getitem__(self, i):

        return self.images[i], self.masks[i]

    def __len__(self):
        return len(self.images)


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),

        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        # albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.GaussNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        # albu.OneOf(
        #     [
        #         albu.IAASharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
    ]
    return albu.Compose(train_transform, is_check_shapes=False)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(384, 480, always_apply=True, border_mode=0),
        # albu.Resize(384, 480)
    ]
    return albu.Compose(test_transform, is_check_shapes=False)


def to_tensor(x, **kwargs):
    return x.astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform, is_check_shapes=False)


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


def parse_filename(filename):
    # Определяем шаблон для поиска
    pattern = r'^(.*?)_LF(\d+)-P(\d+)_(.*?)_(\d+)\.nd2$'

    # Применяем регулярное выражение к имени файла
    match = re.match(pattern, filename)

    if match:
        # Получаем группы из регулярного выражения
        group1 = match.group(1)
        group2 = int(match.group(3))
        group3 = match.group(4)
        group4 = int(match.group(5))

        return group1, group2, group3, group4
    else:
        return None


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


def prepare_data(p, images_num=None, shuffle=True, classes_groups_num=1, allowed_markers=None):
    mask_all = [v for v in os.listdir(p.masks_dir) if v.endswith('.png')]
    
    fn_list = list()
    for fn in os.listdir(p.dataset_dir):
        if fn.endswith('.nd2'):
            if f'{fn}mask.png' in mask_all:
                if allowed_markers is None:
                    fn_list.append(fn)
                else:
                    exp, passage, marker, n = parse_filename(fn)
                    if marker in allowed_markers:
                        fn_list.append(fn)
    fn_list.sort()
    # random.shuffle(fn_list)

    fp = osp.join(p.dataset_dir, fn_list[0])
    np_data = nd2.imread(fp)
    c, w, h = np_data.shape

    print(f'total images: {len(fn_list)}')
    print(f'image shape (c, w, h): {np_data.shape}')

    orig_size = (w, h)
    square_w, square_h = p.square_a, p.square_a
    square_size = (square_w, square_h)

    full_size, full_size_with_borders, squares = get_squares(orig_size,
                                                             square_size,
                                                             p.border)

    # print(f'orig_size: {orig_size}')
    # print(f'full_size: {full_size}')
    # print(f'full_size_with_borders: {full_size_with_borders}')
    # # pprint(core_square_sizes)
    # print(f'squares: {len(squares)}')
    #
    # pprint(squares[:2])
    # print('...')
    # pprint(squares[-2:])

    images_fps = list()
    masks_fps = list()

    for fn in fn_list[:images_num]:
        images_fps.append(osp.join(p.dataset_dir, fn))
        masks_fps.append(osp.join(p.masks_dir, f'{fn}mask.png'))
    images_fps = np.array(images_fps)
    masks_fps = np.array(masks_fps)

    ans = my_train_test_split(images_fps, masks_fps, p.ratio_train, p.ratio_val, shuffle=shuffle)
    X_train, X_val, X_test, y_train, y_val, y_test = ans
    print(X_train.shape, X_val.shape, X_test.shape)

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(p.ENCODER, p.ENCODER_WEIGHTS)
    preprocessing_fn = None
    preprocessing = get_preprocessing(preprocessing_fn)

    if classes_groups_num == 3:
        classes_groups = [[6, 9], [12, 15], [18, 21]]
    else:
        classes_groups = [[6, 9, 12, 15, 18, 21]]
    classes_train = get_classes_from_fps(X_train, classes_groups)
    classes_val = get_classes_from_fps(X_val, classes_groups)
    classes_test = get_classes_from_fps(X_test, classes_groups)
    # classes_train = None
    # classes_val = None
    # classes_test = None
    if p.channels == 4:
        CellDatasetMX = CellDatasetMCnpy
    else:
        CellDatasetMX = CellDatasetMC
    train_dataset = CellDatasetMX(X_train, y_train, squares,
                                  p.border, p.channels, p.classes_num, full_size,
                                  augmentation=get_training_augmentation(),
                                  preprocessing=preprocessing,
                                  classes=classes_train)

    valid_dataset = CellDatasetMX(X_val, y_val, squares,
                                  p.border, p.channels, p.classes_num, full_size,
                                  augmentation=get_validation_augmentation(),
                                  preprocessing=preprocessing,
                                  classes=classes_val)

    test_dataset = CellDatasetMX(X_test, y_test, squares,
                                 p.border, p.channels, p.classes_num, full_size,
                                 augmentation=get_validation_augmentation(),
                                 preprocessing=preprocessing,
                                 classes=classes_test)

    return train_dataset, valid_dataset, test_dataset
