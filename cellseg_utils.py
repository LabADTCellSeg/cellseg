import random
import re
import time
from datetime import datetime
from types import SimpleNamespace
import sys

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as BaseDataset
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils

from concurrent.futures import ThreadPoolExecutor, as_completed

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
        img_sq = image[sq[param_name][1]:sq[param_name][3], sq[param_name][0]:sq[param_name][2], :].copy()
        img_sq_list.append(img_sq)

    return img_sq_list


def split_image(image, full_size, squares, border):
    c = image.shape[-1]

    image_with_borders = []
    for c_idx in range(c):
        image_resized_cur_c = cv2.resize(image[..., c_idx], full_size, interpolation=cv2.INTER_NEAREST)
        image_with_borders_cur_c = cv2.copyMakeBorder(image_resized_cur_c, border, border, border, border,
                                                      cv2.BORDER_REFLECT, None)
        image_with_borders.append(image_with_borders_cur_c)
    image_with_borders = np.stack(image_with_borders, axis=-1)

    img_sq_list = split_on_squares(image_with_borders, squares, 'square_with_borders_coords')

    return image_with_borders, img_sq_list


def unsplit_image(img_sq_list, squares, param_name, border):
    w_num, h_num = squares[-1]['w'] + 1, squares[-1]['h'] + 1
    square_size = (squares[0]['square_coords'][2], squares[0]['square_coords'][3])
    result_size = (img_sq_list[0].shape[0], square_size[0] * h_num, square_size[1] * w_num)
    src_w0, src_w1 = border, square_size[0] + border
    src_h0, src_h1 = border, square_size[1] + border
    result = np.zeros(result_size)
    for sq, img_sq in zip(squares, img_sq_list):
        dst_w0, dst_w1 = sq[param_name][1], sq[param_name][3]
        dst_h0, dst_h1 = sq[param_name][0], sq[param_name][2]
        result[:, dst_w0:dst_w1, dst_h0:dst_h1] = img_sq[:, src_w0:src_w1, src_h0:src_h1]
    return result


def draw_square(image, sq, color):
    for c_idx in range(len(color)):
        image[c_idx][sq[1]:sq[3], sq[0]] = color[c_idx]
        image[c_idx][sq[1], sq[0]:sq[2]] = color[c_idx]
        image[c_idx][sq[1]:sq[3], sq[2]] = color[c_idx]
        image[c_idx][sq[3], sq[0]:sq[2]] = color[c_idx]


def my_train_test_split(x, y, ratio_train, ratio_val, shuffle=True, seed=42):
    idx = np.arange(x.shape[0])
    np.random.seed(seed)
    if shuffle:
        np.random.shuffle(idx)

    limit_train = int(ratio_train * x.shape[0])
    limit_val = int((ratio_train + ratio_val) * x.shape[0])

    idx_train = idx[:limit_train]
    idx_val = idx[limit_train:limit_val]
    idx_test = idx[limit_val:]

    x_train, y_train = x[idx_train], y[idx_train]
    x_val, y_val = x[idx_val], y[idx_val]
    x_test, y_test = x[idx_test], y[idx_test]

    return x_train, x_val, x_test, y_train, y_val, y_test


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
                               idx=idx,
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
            full_size=None,
            add_shadow_to_img=False,
            squares=None,
            border=None,
            classes=None,
            augmentation=None,
            preprocessing=None,
            target_size=None,
            contour_thickness=2,
            max_workers=8
    ):
        self.all_fp_data = all_fp_data
        self.full_size = full_size

        self.add_shadow_to_img = add_shadow_to_img

        self.squares = squares
        self.border = border
        self.channels = None

        self.classes = classes

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.max_workers = max_workers

        self.target_size = target_size

        self.contour_thickness = contour_thickness

        if self.border is None:
            self.border = 0

        self.images = list()
        self.masks = list()
        self.shadows = list()
        self.info = list()

        if self.max_workers <= 0:
            self.default_processing()
        else:
            self.parallel_processing(max_workers=self.max_workers)  # запуск параллельной обработки

        if len(self.images) != 0:
            self.channels = self.images[0].shape[-1]
            if self.add_shadow_to_img:
                self.channels += 1

    def process_fp_data(self, fp_data):
        """Метод для обработки одного объекта данных."""
        img = self.read_image(fp_data)
        mask = self.read_mask(fp_data) // 255
        assert img.shape[:-1] == mask.shape[:-1]

        if self.classes is not None:
            mask = self._prepare_mask(mask, cls_num=len(self.classes), cls=self.classes.index(fp_data['cls']),
                                      contour_thickness=self.contour_thickness)
        else:
            mask = self._prepare_mask(mask, cls_num=1, cls=0, contour_thickness=self.contour_thickness)
        shadow = self.read_shadow(fp_data)[..., 0:1] if self.add_shadow_to_img else None

        result_images, result_masks, result_shadows, result_info = [], [], [], []

        if all(v is not None for v in [self.full_size, self.squares, self.border]):
            # Разбиение на части
            _, img_sq_list = split_image(img, self.full_size, self.squares, self.border)
            _, mask_sq_list = split_image(mask, self.full_size, self.squares, self.border)

            if self.add_shadow_to_img:
                _, shadow_sq_list = split_image(shadow, self.full_size, self.squares, self.border)
            else:
                shadow_sq_list = [None] * len(img_sq_list)

            for img_sq, msk_sq, shd_sq, sq in zip(img_sq_list, mask_sq_list, shadow_sq_list, self.squares):
                result_images.append(img_sq)
                result_masks.append(msk_sq.astype(np.bool_))
                result_shadows.append(shd_sq)

                squares_info = dict(sq=sq)
                squares_info.update(fp_data)
                result_info.append(squares_info)
        else:
            result_images.append(img)
            result_masks.append(mask.astype(np.bool_))
            result_shadows.append(shadow)
            result_info.append(fp_data)

        return result_images, result_masks, result_shadows, result_info

    def default_processing(self):
        for fp_data in tqdm(self.all_fp_data):
            images, masks, shadows, info = self.process_fp_data(fp_data)
            self.images.extend(images)
            self.masks.extend(masks)
            self.shadows.extend(shadows)
            self.info.extend(info)

    def parallel_processing(self, max_workers=4):
        """Параллельная обработка данных."""
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_fp_data, fp_data): fp_data for fp_data in self.all_fp_data}
            for future in tqdm(as_completed(futures), total=len(futures)):
                try:
                    images, masks, shadows, info = future.result()
                    self.images.extend(images)
                    self.masks.extend(masks)
                    self.shadows.extend(shadows)
                    self.info.extend(info)
                except Exception as e:
                    print(f"Ошибка при обработке данных: {e}")

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        if channels is None:
            channels = ['r', 'g', 'b']

        c_stack = list()
        for c_idx, c in enumerate(channels):
            c_img = Image.open(fp_data[f'{c}_fp'])

            # if self.target_size:
            #     if self.target_size[0] != c_img.size[0] or self.target_size[1] != c_img.size[1]:
            #         c_img = c_img.resize(self.target_size)
            # else:
            #     new_width = c_img.size[0] // 32 * 32
            #     new_height = c_img.size[1] // 32 * 32
            #     if new_width != c_img.size[0] or new_height != c_img.size[1]:
            #         c_img = c_img.resize((new_width, new_height))

            if convert:
                c_img = np.asarray(c_img.convert(convert))
            else:
                c_img = np.asarray(c_img)[..., c_idx]
            c_stack.append(c_img)

        img_np = np.stack(c_stack, axis=-1)

        return img_np

    @staticmethod
    def read_mask(fp_data):
        return CellDataset4.read_image(fp_data, channels=['mask'])

    @staticmethod
    def read_shadow(fp_data):
        return CellDataset4.read_image(fp_data, channels=['p'], convert='L')

    @staticmethod
    def _prepare_mask(mask, cls_num=1, cls=0, contour_thickness=2):
        mask_contour = mask.copy()
        contours, _ = cv2.findContours(mask_contour,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, contourIdx=-1, color=[2], thickness=contour_thickness)
        # mask_contour = mask_contour.transpose(2, 0, 1)
        # mask_contour = mask_contour
        mask = np.zeros((mask.shape[0], mask.shape[1], cls_num + 1), dtype=mask.dtype)
        mask[..., cls:cls + 1][mask_contour == 1] = 1
        mask[..., cls_num:cls_num + 1][mask_contour == 2] = 1
        # mask = np.zeros((mask.shape[0], mask.shape[1], 2), dtype=mask.dtype)
        # mask[..., 0:1][mask_contour == 1] = 1
        # mask[..., 1:2][mask_contour == 2] = 1

        return mask

    def aug(self, img, mask, shadow=False):
        if self.augmentation or self.preprocessing:
            # img_orig_dtype = img.dtype
            # mask_orig_dtype = mask.dtype

            mask = mask.astype(np.uint8)

            if shadow is None:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']

                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']

                # img = img.astype(img_orig_dtype)
                # mask = mask.astype(mask_orig_dtype)

            else:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']

                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']

                # shadow = np.asarray(Image.fromarray(np.dstack[np.uint8(shadow)]*3)).convert('L'))
                # shadow = shadow[..., 0].astype(img_orig_dtype)
                # img = (img * 255).astype(img_orig_dtype)
                img = np.dstack([img, shadow[..., 0]])
                # mask = mask.astype(mask_orig_dtype)

        return img / 255, mask

    def __getitem__(self, i):
        img, mask = self.aug(self.images[i], self.masks[i], shadow=self.shadows[i])

        return img.transpose(2, 0, 1).astype(np.float32), mask.transpose(2, 0, 1).astype(np.float32)

    def __len__(self):
        return len(self.images)


class ApplyToImageOnly(A.ImageOnlyTransform):
    def __init__(self, transform, always_apply=False, p=1.0):
        super(ApplyToImageOnly, self).__init__(always_apply, p)
        self.transform = transform

    def apply(self, img, **params):
        # Проверяем, что это основное изображение, а не shadow
        if img.shape[2] == 3:  # Только для трехканальных изображений
            return self.transform(image=img)["image"]
        return img


def get_training_augmentation(target_size=None):
    train_transform = [

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        A.ShiftScaleRotate(shift_limit=0, scale_limit=0.25, rotate_limit=0, p=1, border_mode=0),

        A.RandomResizedCrop(height=target_size[1], width=target_size[0], scale=(0.75, 1.0), p=0.5),

        A.GaussNoise(p=0.2),

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
        ApplyToImageOnly(A.HueSaturationValue(p=0.9)),  # Применяем только к основному изображению
    ]
    if target_size is not None:
        train_transform.append(A.Resize(height=target_size[1], width=target_size[0], always_apply=True))

    return A.Compose(train_transform, additional_targets={'shadow': 'image'})


def get_validation_augmentation(target_size=None):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # A.PadIfNeeded(384, 480, always_apply=True, border_mode=0),
        # A.Resize(384, 480)
    ]
    if target_size is not None:
        test_transform.append(A.Resize(height=target_size[1], width=target_size[0], always_apply=True))
    return A.Compose(test_transform, is_check_shapes=False, additional_targets={'shadow': 'image'})


def to_tensor(x, **kwargs):
    return x.astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: Augmentations.Compose

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
        if self.bce_weight == 0:
            self.__call__ = self.dice_loss_calc
        elif self.bce_weight == 1:
            self.__call__ = self.bce_loss_calc

    def __call__(self, pred, target):
        return (self.bce_loss_calc(pred, target) * self.bce_weight +
                self.dice_loss_calc(pred, target) * (1 - self.bce_weight))

    def dice_loss_calc(self, pred, target):
        return dice_loss(torch.sigmoid(pred), target).to(self.device)

    def bce_loss_calc(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean').to(self.device)

    def to(self, device):
        self.device = device


class WeightedBCEDiceLoss:
    __name__ = 'bce_dice'

    def __init__(self, bce_weight=0.5, boundary_weight=0.99):
        self.bce_weight = bce_weight
        self.device = 'cpu'
        self.boundary_weight = boundary_weight
        if self.bce_weight == 0:
            self.__call__ = self.dice_loss_calc
        elif self.bce_weight == 1:
            self.__call__ = self.bce_loss_calc

    def __call__(self, pred, target):
        # Рассчитываем BCE и Dice для каждого канала
        bce_cells = self.bce_loss_calc(pred[:, 0, :, :], target[:, 0, :, :])
        bce_boundaries = self.bce_loss_calc(pred[:, 1, :, :], target[:, 1, :, :])

        dice_cells = self.dice_loss_calc(torch.sigmoid(pred[:, 0, :, :]), target[:, 0, :, :])
        dice_boundaries = self.dice_loss_calc(torch.sigmoid(pred[:, 1, :, :]), target[:, 1, :, :])

        # Применяем разные веса для границ
        bce_loss = (bce_cells + self.boundary_weight * bce_boundaries) / (1 + self.boundary_weight)
        dice_loss = (dice_cells + self.boundary_weight * dice_boundaries) / (1 + self.boundary_weight)

        return bce_loss * self.bce_weight + dice_loss * (1 - self.bce_weight)

    def dice_loss_calc(self, pred, target):
        return dice_loss(torch.sigmoid(pred), target).to(self.device)

    def bce_loss_calc(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target, reduction='mean').to(self.device)

    def to(self, device):
        self.device = device


def parse_filename_nd2(filename):
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
        exp, p, marker, n = parse_filename_nd2(fn)
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


def prepare_data_from_params(params, shuffle=True, max_workers=8):
    return prepare_data(dataset_dir=params.dataset_dir,
                        exp_class_dict=params.exp_class_dict,
                        square_a=params.square_a,
                        border=params.border,
                        ratio_train=params.ratio_train,
                        ratio_val=params.ratio_val,
                        images_num=params.images_num,
                        multiclass=params.multiclass,
                        add_shadow_to_img=params.add_shadow_to_img,
                        contour_thickness=params.contour_thickness,
                        shuffle=shuffle,
                        max_workers=max_workers)


def prepare_data(dataset_dir,
                 exp_class_dict,
                 square_a=236,
                 border=10,
                 ratio_train=0.6,
                 ratio_val=0.2,
                 images_num=None,
                 shuffle=True,
                 resize_coef=1,
                 multiclass=False,
                 add_shadow_to_img=True,
                 contour_thickness=2,
                 max_workers=8):
    all_fp_data = get_all_fp_data(dataset_dir, exp_class_dict)
    if shuffle:
        random.Random(42).shuffle(all_fp_data)
    all_fp_data = all_fp_data[:images_num]

    total_len = len(all_fp_data)
    train_num = int(total_len * ratio_train)
    val_num = int(total_len * ratio_val)
    # test_num = total_len - val_num

    train_fp_data = all_fp_data[:train_num]
    val_fp_data = all_fp_data[train_num:train_num + val_num]
    test_fp_data = all_fp_data[train_num + val_num:]

    img_path = all_fp_data[0]['mask_fp']
    with Image.open(img_path) as img:
        new_width = int(img.size[0] / resize_coef) // 32 * 32
        new_height = int(img.size[1] / resize_coef) // 32 * 32
        target_size = (new_width, new_height)

    if square_a is None:
        full_size, squares = None, None
    else:
        square_w, square_h = square_a, square_a
        square_size = (square_w, square_h)
        full_size, full_size_with_borders, squares = get_squares(target_size, square_size, border)

    # preprocessing_fn = smp.encoders.get_preprocessing_fn(p.ENCODER, p.ENCODER_WEIGHTS)
    preprocessing_fn = None
    preprocessing = get_preprocessing(preprocessing_fn)

    transform_size = squares[0]['square_with_borders_coords'][2:]

    if multiclass:
        classes = list(set([v for v in exp_class_dict.values()]))
        classes.sort()

    else:
        classes = None

    def dataset_fn(fp_data, aug):
        return CellDataset4(fp_data,
                            full_size=full_size,
                            add_shadow_to_img=add_shadow_to_img,
                            squares=squares,
                            border=border,
                            augmentation=aug,
                            preprocessing=preprocessing,
                            classes=classes,
                            target_size=target_size,
                            contour_thickness=contour_thickness,
                            max_workers=max_workers
                            )

    fp_data_list = SimpleNamespace(train=train_fp_data,
                                   valid=val_fp_data,
                                   test=test_fp_data)
    aug_list = SimpleNamespace(train=get_training_augmentation(target_size=transform_size),
                               valid=get_validation_augmentation(target_size=transform_size),
                               test=get_validation_augmentation(target_size=transform_size))

    return fp_data_list, aug_list, dataset_fn


class TrainEpochSchedulerStep(smp.utils.train.Epoch):
    def __init__(self, model, loss, metrics, optimizer, scheduler, scheduler_step_every_batch=False,
                 device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_every_batch = scheduler_step_every_batch

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        if self.scheduler_step_every_batch:
            self.scheduler.step()
        return loss, prediction

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = smp.utils.train.AverageValueMeter()
        metrics_meters = {metric.__name__: smp.utils.train.AverageValueMeter() for metric in self.metrics}

        with tqdm(
                dataloader,
                desc=self.stage_name,
                file=sys.stdout,
                disable=not self.verbose,
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                lr_logs = {'LR': self.scheduler.get_last_lr()[0]}
                logs.update(lr_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs
