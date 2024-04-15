import time
from datetime import datetime

import nd2
import numpy as np

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

    square_w_num, square_h_num = int(np.ceil(w/square_w)), int(np.ceil(h/square_h))
    full_size = (square_w*square_w_num, square_h*square_h_num)
    full_size_with_borders = (full_size[0]+2*border, full_size[1]+2*border)

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
                                          end_w+2*border,
                                          end_h+2*border]
            square_with_borders_coords_rev = [start_w+border,
                                              start_h+border,
                                              end_w+border,
                                              end_h+border]

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
        image_with_borders_cur_c = cv2.copyMakeBorder(image_resized_cur_c, border, border, border, border, cv2.BORDER_REFLECT, None)
        image_with_borders.append(image_with_borders_cur_c)
    image_with_borders = np.stack(image_with_borders)

    img_sq_list = split_on_squares(image_with_borders, squares, 'square_with_borders_coords')

    return image_with_borders, img_sq_list


def unsplit_image(img_sq_list, squares, param_name, border):
    w_num, h_num = squares[-1]['w']+1, squares[-1]['w']+1
    square_size = (squares[0]['square_coords'][2], squares[0]['square_coords'][3])
    result_size = (img_sq_list[0].shape[0], square_size[0]*w_num, square_size[1]*h_num)
    result = np.zeros(result_size)
    for sq, img_sq in zip(squares, img_sq_list):
        result[:, sq[param_name][1]:sq[param_name][3], sq[param_name][0]:sq[param_name][2]] = img_sq[:, border:square_size[0]+border, border:square_size[1]+border]
    return result


def draw_square(image, sq, color):
    for c_idx in range(len(color)):
        image[c_idx][sq[1]:sq[3], sq[0]] = color[c_idx]
        image[c_idx][sq[1], sq[0]:sq[2]] = color[c_idx]
        image[c_idx][sq[1]:sq[3], sq[2]] = color[c_idx]
        image[c_idx][sq[3], sq[0]:sq[2]] = color[c_idx]


def my_train_test_split(X, y, ratio_train, ratio_val, seed=42):
    idx = np.arange(X.shape[0])
    np.random.seed(seed)
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


class CellDataset(BaseDataset):
    def __init__(
            self,
            images_fps,
            masks_fps,
            squares,
            border,
            channels,
            classes,
            full_size,
            augmentation=None,
            preprocessing=None,
    ):
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        self.squares = squares
        self.border = border
        self.channels = channels
        self.classes = classes
        self.full_size = full_size

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.images, self.masks, self.squares_info = self._create_square()

    def _create_square(self):
        images = list()
        masks = list()
        squares_info = list()
        for fp, mask_fp in tqdm(zip(self.images_fps, self.masks_fps)):
            img = nd2.imread(fp)
            img = img / 4095
            mask = cv2.imread(mask_fp, cv2.IMREAD_GRAYSCALE)[np.newaxis, ...]
            mask = mask / 255

            mask_contour = mask.copy()
            mask_contour = mask_contour.transpose(1, 2, 0).astype('uint8')
            mask2 = np.zeros_like(mask_contour)
            contours, hierarchy = cv2.findContours(mask_contour,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            _ = cv2.drawContours(mask2, contours, -1, 1, 2)
            mask2 = mask2.transpose(2, 0, 1).astype('float64')
            # print(mask.shape, mask.dtype)
            # print(mask2.shape, mask2.dtype)
            mask = np.vstack([mask, mask2])
            # print(mask.shape)
            assert img.shape == (2, 1024, 1024)
            assert mask.shape == (2, 1024, 1024)

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

                images.append(img_sq)
                masks.append(msk_sq)
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

        albu.IAAAdditiveGaussianNoise(p=0.2),
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