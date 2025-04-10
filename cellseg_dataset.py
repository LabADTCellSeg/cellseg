import random
from types import SimpleNamespace
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import cv2
import numpy as np

from torch.utils.data import Dataset as BaseDataset
import albumentations as A

from cellseg_utils import get_all_fp_data


class CellDataset4(BaseDataset):
    def __init__(
            self,
            all_fp_data,
            full_size=None,
            add_shadow_to_img=False,
            squares=None,
            border=None,
            classes=None,
            channels=['r', 'g', 'b'],
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
        self.channels = channels
        self.channels_num = None
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
            self.channels_num = self.images[0].shape[-1]
            if self.add_shadow_to_img:
                self.channels_num += 1

    def process_fp_data(self, fp_data):
        """Метод для обработки одного объекта данных."""
        img = self.read_image(fp_data, channels=self.channels)
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


class CellDataset4Test1(BaseDataset):
    def __init__(
            self,
            all_fp_data,
            full_size=None,
            add_shadow_to_img=False,
            squares=None,
            border=None,
            classes=None,
            channels=['r', 'g', 'b'],
            augmentation=None,
            preprocessing=None,
            target_size=None,
            contour_thickness=2
    ):
        """
        Параметры аналогичны оригинальному датасету, но здесь данные не предвычисляются.
        Построение mapping-а по индексам позволяет динамически загружать нужный квадрат.
        """
        self.all_fp_data = all_fp_data
        self.full_size = full_size
        self.add_shadow_to_img = add_shadow_to_img
        self.squares = squares
        self.border = border if border is not None else 0
        self.classes = classes
        self.channels = channels
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.target_size = target_size
        self.contour_thickness = contour_thickness

        # Построим отображение: каждому глобальному индексу датасета соответствует пара:
        # (индекс fp_data, индекс квадрата внутри изображения).
        self.mapping = []
        for fp_idx, fp in enumerate(self.all_fp_data):
            if self.full_size is not None and self.squares is not None:
                # Если производится разбиение на квадраты, добавляем столько записей,
                # сколько квадратов задано в списке squares.
                for sq_idx, _ in enumerate(self.squares):
                    self.mapping.append((fp_idx, sq_idx))
            else:
                # Если разбиение не требуется – одна запись на изображение.
                self.mapping.append((fp_idx, None))

        # Для ускорения последовательного обхода: кэш последнего загруженного файла.
        self._cached_fp_index = None
        self._cached_data = None  # Будет содержать кортеж (images, masks, shadows, info)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        """
        При запросе элемента определяется, к какому файлу и квадрату он относится.
        Если нужный файл уже кэширован – используется кэш, иначе происходит загрузка.
        После извлечения последнего квадрата файла кэш очищается, освобождая память.
        """
        fp_idx, square_idx = self.mapping[index]

        # Если кэширован именно нужный файл, используем данные из кэша.
        if self._cached_fp_index == fp_idx:
            images, masks, shadows, info = self._cached_data
        else:
            fp_data = self.all_fp_data[fp_idx]
            images, masks, shadows, info = self.process_fp_data(fp_data)
            self._cached_fp_index = fp_idx
            self._cached_data = (images, masks, shadows, info)

        # Выбираем нужный квадрат или всё изображение, если разбиение не производится.
        if square_idx is not None:
            img = images[square_idx]
            mask = masks[square_idx]
            shadow = shadows[square_idx]
        else:
            img = images[0]
            mask = masks[0]
            shadow = shadows[0]

        # Применяем аугментацию/препроцессинг, если они заданы
        img_aug, mask_aug = self.aug(img, mask, shadow=shadow)
        result = (img_aug.transpose(2, 0, 1).astype(np.float32),
                  mask_aug.transpose(2, 0, 1).astype(np.float32))

        # Механизм освобождения памяти: если текущий квадрат – последний для данного файла,
        # либо это последний элемент всего датасета, очищаем кэш.
        if index == len(self.mapping) - 1 or self.mapping[index + 1][0] != fp_idx:
            self._cached_fp_index = None
            self._cached_data = None

        return result

    def process_fp_data(self, fp_data):
        """
        Загружает изображение и маску, выполняет предварительную обработку
        и, при необходимости, разбивает изображение на квадраты.
        """
        img = self.read_image(fp_data, channels=self.channels)
        mask = self.read_mask(fp_data) // 255
        assert img.shape[:-1] == mask.shape[:-1]

        if self.classes is not None:
            mask = self._prepare_mask(mask, cls_num=len(self.classes),
                                      cls=self.classes.index(fp_data['cls']),
                                      contour_thickness=self.contour_thickness)
        else:
            mask = self._prepare_mask(mask, cls_num=1, cls=0,
                                      contour_thickness=self.contour_thickness)
        shadow = self.read_shadow(fp_data)[..., 0:1] if self.add_shadow_to_img else None

        result_images, result_masks, result_shadows, result_info = [], [], [], []

        if all(v is not None for v in [self.full_size, self.squares, self.border]):
            # Разбиваем изображение и маску на квадраты.
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
            # Если не задано разбиение на квадраты, возвращается всё изображение.
            result_images.append(img)
            result_masks.append(mask.astype(np.bool_))
            result_shadows.append(shadow)
            result_info.append(fp_data)

        return result_images, result_masks, result_shadows, result_info

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """
        Считывание изображения по указанным каналам.
        """
        if channels is None:
            channels = ['r', 'g', 'b']

        c_stack = []
        for c_idx, c in enumerate(channels):
            c_img = Image.open(fp_data[f'{c}_fp'])
            if convert:
                c_img = np.asarray(c_img.convert(convert))
            else:
                # Если не требуется конвертация, извлекаем канал по индексу.
                c_img = np.asarray(c_img)[..., c_idx]
            c_stack.append(c_img)
        return np.stack(c_stack, axis=-1)

    @staticmethod
    def read_mask(fp_data):
        return CellDataset4Test.read_image(fp_data, channels=['mask'])

    @staticmethod
    def read_shadow(fp_data):
        return CellDataset4Test.read_image(fp_data, channels=['p'], convert='L')

    @staticmethod
    def _prepare_mask(mask, cls_num=1, cls=0, contour_thickness=2):
        """
        Подготавливает маску: находит контуры и создаёт дополнительный канал для контура.
        """
        mask_contour = mask.copy()
        contours, _ = cv2.findContours(mask_contour,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, contourIdx=-1,
                             color=[2], thickness=contour_thickness)
        mask_prepared = np.zeros((mask.shape[0], mask.shape[1], cls_num + 1),
                                 dtype=mask.dtype)
        mask_prepared[..., cls:cls + 1][mask_contour == 1] = 1
        mask_prepared[..., cls_num:cls_num + 1][mask_contour == 2] = 1
        return mask_prepared

    def aug(self, img, mask, shadow=False):
        """
        Применяет аугментацию и/или препроцессинг, если они заданы.
        Если shadow передан, добавляет его как дополнительный канал.
        """
        if self.augmentation or self.preprocessing:
            mask = mask.astype(np.uint8)
            if shadow is None:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']
                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']
            else:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']
                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']
                # Добавляем канал тени (при условии, что shadow имеет 1 канал)
                img = np.dstack([img, shadow[..., 0]])
            return img / 255, mask
        else:
            return img / 255, mask


class CellDataset4Test2(BaseDataset):
    def __init__(
            self,
            all_fp_data,
            full_size=None,
            add_shadow_to_img=False,
            squares=None,
            border=None,
            classes=None,
            channels=['r', 'g', 'b'],
            augmentation=None,
            preprocessing=None,
            target_size=None,
            contour_thickness=2
    ):
        """
        Параметры аналогичны оригинальному датасету, но здесь данные не предвычисляются.
        Построение mapping-а по индексам позволяет динамически загружать нужный квадрат.
        Реализован кэш с поддержкой нескольких файлов одновременно и механикой освобождения памяти,
        когда все квадраты файла запрошены.
        """
        self.all_fp_data = all_fp_data
        self.full_size = full_size
        self.add_shadow_to_img = add_shadow_to_img
        self.squares = squares
        self.border = border if border is not None else 0
        self.classes = classes
        self.channels = channels
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.target_size = target_size
        self.contour_thickness = contour_thickness

        # Построим отображение: каждому глобальному индексу датасета соответствует пара:
        # (индекс fp_data, индекс квадрата внутри изображения).
        self.mapping = []
        # Также для каждого fp_data узнаем, сколько квадратиков должно быть запрошено.
        self.file_total_counts = {}
        for fp_idx, fp in enumerate(self.all_fp_data):
            if self.full_size is not None and self.squares is not None:
                num_sq = len(self.squares)
                for sq_idx in range(num_sq):
                    self.mapping.append((fp_idx, sq_idx))
                self.file_total_counts[fp_idx] = num_sq
            else:
                self.mapping.append((fp_idx, None))
                self.file_total_counts[fp_idx] = 1

        # Используем словарь для кэширования, чтобы поддерживать сразу несколько загруженных fp.
        self._cache = {}         # ключ: fp_index, значение: (images, masks, shadows, info)
        self._cache_usage = {}   # ключ: fp_index, значение: число уже выданных квадратов

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        """
        По запросу элемента определяется, к какому файлу и какому квадрату он относится.
        Если данные для нужного файла уже в кэше, используется кэш; иначе производится загрузка.
        После возвращения элемента обновляется счётчик использования для соответствующего файла,
        и если все квадраты этого файла выданы, данные удаляются из кэша.
        """
        fp_idx, square_idx = self.mapping[index]

        # Загружаем данные из кэша, если они уже есть для данного fp_idx.
        if fp_idx in self._cache:
            images, masks, shadows, info = self._cache[fp_idx]
        else:
            fp_data = self.all_fp_data[fp_idx]
            images, masks, shadows, info = self.process_fp_data(fp_data)
            self._cache[fp_idx] = (images, masks, shadows, info)
            self._cache_usage[fp_idx] = 0

        # Выбираем нужный квадрат (если разбиение применяется) или всё изображение.
        if square_idx is not None:
            img = images[square_idx]
            mask = masks[square_idx]
            shadow = shadows[square_idx]
        else:
            img = images[0]
            mask = masks[0]
            shadow = shadows[0]

        # Применяем аугментацию/препроцессинг, если они заданы
        img_aug, mask_aug = self.aug(img, mask, shadow=shadow)
        result = (img_aug.transpose(2, 0, 1).astype(np.float32),
                  mask_aug.transpose(2, 0, 1).astype(np.float32))

        # Обновляем счетчик использования для данного файла
        self._cache_usage[fp_idx] += 1
        # Если запрошено все квадраты из этого файла, освобождаем кэш
        if self._cache_usage[fp_idx] >= self.file_total_counts[fp_idx]:
            del self._cache[fp_idx]
            del self._cache_usage[fp_idx]

        return result

    def process_fp_data(self, fp_data):
        """
        Загружает изображение и маску, выполняет предварительную обработку,
        а при необходимости – разбивает на квадраты.
        """
        img = self.read_image(fp_data, channels=self.channels)
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
            # Разбиваем изображение и маску на квадраты.
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

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """
        Считывание изображения по указанным каналам.
        """
        if channels is None:
            channels = ['r', 'g', 'b']

        c_stack = []
        for c_idx, c in enumerate(channels):
            c_img = Image.open(fp_data[f'{c}_fp'])
            if convert:
                c_img = np.asarray(c_img.convert(convert))
            else:
                c_img = np.asarray(c_img)[..., c_idx]
            c_stack.append(c_img)
        return np.stack(c_stack, axis=-1)

    @staticmethod
    def read_mask(fp_data):
        return CellDataset4Test.read_image(fp_data, channels=['mask'])

    @staticmethod
    def read_shadow(fp_data):
        return CellDataset4Test.read_image(fp_data, channels=['p'], convert='L')

    @staticmethod
    def _prepare_mask(mask, cls_num=1, cls=0, contour_thickness=2):
        """
        Подготавливает маску: выделяет контуры и создаёт дополнительный канал для контура.
        """
        mask_contour = mask.copy()
        contours, _ = cv2.findContours(mask_contour,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, contourIdx=-1,
                             color=[2], thickness=contour_thickness)
        mask_prepared = np.zeros((mask.shape[0], mask.shape[1], cls_num + 1),
                                 dtype=mask.dtype)
        mask_prepared[..., cls:cls + 1][mask_contour == 1] = 1
        mask_prepared[..., cls_num:cls_num + 1][mask_contour == 2] = 1
        return mask_prepared

    def aug(self, img, mask, shadow=False):
        """
        Применяет аугментацию и/или препроцессинг, если они заданы.
        Если передан shadow, добавляет его как дополнительный канал.
        """
        if self.augmentation or self.preprocessing:
            mask = mask.astype(np.uint8)
            if shadow is None:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']
                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']
            else:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']
                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']
                # Добавление канала тени (если shadow имеет 1 канал)
                img = np.dstack([img, shadow[..., 0]])
            return img / 255, mask
        else:
            return img / 255, mask


class CellDataset4Test(BaseDataset):
    def __init__(
            self,
            all_fp_data,
            full_size=None,
            add_shadow_to_img=False,
            squares=None,
            border=None,
            classes=None,
            channels=['r', 'g', 'b'],
            augmentation=None,
            preprocessing=None,
            target_size=None,
            contour_thickness=2
    ):
        """
        Датасет для тестирования, который динамически загружает и обрабатывает данные.
        При использовании batch_size > 1 реализовано кэширование нескольких файлов с
        освобождением памяти по файлу, когда все его квадраты запрошены.
        Дополнительно сохраняется info для каждого элемента, чтобы его можно было использовать
        при отрисовке результатов.
        """
        self.all_fp_data = all_fp_data
        self.full_size = full_size
        self.add_shadow_to_img = add_shadow_to_img
        self.squares = squares
        self.border = border if border is not None else 0
        self.classes = classes
        self.channels = channels
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.target_size = target_size
        self.contour_thickness = contour_thickness

        # Формируем mapping: для каждого глобального индекса записываем (fp_index, индекс квадрата)
        self.mapping = []
        for fp_idx, fp in enumerate(self.all_fp_data):
            if self.full_size is not None and self.squares is not None:
                for sq_idx in range(len(self.squares)):
                    self.mapping.append((fp_idx, sq_idx))
            else:
                self.mapping.append((fp_idx, None))

        # Создаем список для хранения info для каждого элемента.
        self.info = [None] * len(self.mapping)

        # Создаем словарь для быстрого доступа: для каждого fp_idx список индексов в mapping
        self.mapping_indices = {}
        for idx, (fp_idx, square_idx) in enumerate(self.mapping):
            if fp_idx not in self.mapping_indices:
                self.mapping_indices[fp_idx] = []
            self.mapping_indices[fp_idx].append(idx)

        # Словарь для кэширования данных файлов.
        # Для каждого fp_index будем хранить: (images, masks, shadows, info, expected_count)
        self._cache = {}
        # Счетчик выданных элементов для каждого fp_index.
        self._cache_usage = {}

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        """
        По индексу возвращает картинку и маску, при этом:
         - Если данные файла уже кэшированы, используются кэшированные данные.
         - После выдачи требуемого числа квадратов для файла, кэш удаляется.
         - Поле info сохраняется в self.info для использования в отрисовке.
        """
        fp_idx, square_idx = self.mapping[index]

        # Если данные файла уже кэшированы — используем их.
        if fp_idx in self._cache:
            images, masks, shadows, infos, expected_count = self._cache[fp_idx]
        else:
            fp_data = self.all_fp_data[fp_idx]
            images, masks, shadows, infos = self.process_fp_data(fp_data)
            expected_count = len(images)  # реальное число полученных квадратов
            self._cache[fp_idx] = (images, masks, shadows, infos, expected_count)
            self._cache_usage[fp_idx] = 0

            # Заполняем сразу все элементы self.info, соответствующие данному fp_idx.
            for i in self.mapping_indices.get(fp_idx, []):
                _, sq = self.mapping[i]
                self.info[i] = infos[sq] if sq is not None else infos[0]

        # Если используется разбиение на квадраты, выбираем нужный квадрат, иначе берём первый элемент.
        if square_idx is not None:
            img = images[square_idx]
            mask = masks[square_idx]
            shadow = shadows[square_idx]
            info = infos[square_idx]
        else:
            img = images[0]
            mask = masks[0]
            shadow = shadows[0]
            info = infos[0]

        # Применяем аугментацию/препроцессинг, если заданы.
        img_aug, mask_aug = self.aug(img, mask, shadow=shadow)
        result = (img_aug.transpose(2, 0, 1).astype(np.float32),
                  mask_aug.transpose(2, 0, 1).astype(np.float32))

        # Можно обновить self.info для текущего индекса (если требуется перезапись)
        self.info[index] = info

        # Обновляем счетчик использования для данного файла.
        self._cache_usage[fp_idx] += 1
        # Если для данного файла выдано все ожидаемые элементы – освобождаем кэш.
        if self._cache_usage[fp_idx] >= expected_count:
            del self._cache[fp_idx]
            del self._cache_usage[fp_idx]

        return result

    def process_fp_data(self, fp_data):
        """
        Загружает изображение и маску, выполняет предварительную обработку и, при необходимости,
        разбивает изображение на квадраты. Помимо изображений и масок, возвращает соответствующую
        info (словарь, содержащий поля из fp_data и параметр 'sq').
        """
        img = self.read_image(fp_data, channels=self.channels)
        mask = self.read_mask(fp_data) // 255
        assert img.shape[:-1] == mask.shape[:-1]

        if self.classes is not None:
            mask = self._prepare_mask(mask, cls_num=len(self.classes),
                                      cls=self.classes.index(fp_data['cls']),
                                      contour_thickness=self.contour_thickness)
        else:
            mask = self._prepare_mask(mask, cls_num=1, cls=0,
                                      contour_thickness=self.contour_thickness)
        shadow = self.read_shadow(fp_data)[..., 0:1] if self.add_shadow_to_img else None

        result_images, result_masks, result_shadows, result_infos = [], [], [], []

        if all(v is not None for v in [self.full_size, self.squares, self.border]):
            # Разбиваем изображение и маску на квадраты.
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
                # Формируем info: включаем координаты квадрата (или другие поля) и поля из fp_data.
                info_sq = dict(sq=sq)
                info_sq.update(fp_data)
                result_infos.append(info_sq)
        else:
            result_images.append(img)
            result_masks.append(mask.astype(np.bool_))
            result_shadows.append(shadow)
            result_infos.append(fp_data)

        return result_images, result_masks, result_shadows, result_infos

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """
        Считывает изображение по заданным каналам.
        """
        if channels is None:
            channels = ['r', 'g', 'b']
        c_stack = []
        for c_idx, c in enumerate(channels):
            c_img = Image.open(fp_data[f'{c}_fp'])
            if convert:
                c_img = np.asarray(c_img.convert(convert))
            else:
                c_img = np.asarray(c_img)[..., c_idx]
            c_stack.append(c_img)
        return np.stack(c_stack, axis=-1)

    @staticmethod
    def read_mask(fp_data):
        return CellDataset4Test.read_image(fp_data, channels=['mask'])

    @staticmethod
    def read_shadow(fp_data):
        return CellDataset4Test.read_image(fp_data, channels=['p'], convert='L')

    @staticmethod
    def _prepare_mask(mask, cls_num=1, cls=0, contour_thickness=2):
        """
        Выделяет контуры на маске и создает дополнительный канал для контура.
        """
        mask_contour = mask.copy()
        contours, _ = cv2.findContours(mask_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, contourIdx=-1, color=[2], thickness=contour_thickness)
        mask_prepared = np.zeros((mask.shape[0], mask.shape[1], cls_num + 1), dtype=mask.dtype)
        mask_prepared[..., cls:cls + 1][mask_contour == 1] = 1
        mask_prepared[..., cls_num:cls_num + 1][mask_contour == 2] = 1
        return mask_prepared

    def aug(self, img, mask, shadow=False):
        """
        Если заданы аугментация или препроцессинг, то применяет их.
        При наличии shadow добавляет его как дополнительный канал.
        """
        if self.augmentation or self.preprocessing:
            mask = mask.astype(np.uint8)
            if shadow is None:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']
                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask)
                    img, mask = sample['image'], sample['mask']
            else:
                if self.augmentation:
                    sample = self.augmentation(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']
                if self.preprocessing:
                    sample = self.preprocessing(image=img, mask=mask, shadow=shadow)
                    img, mask, shadow = sample['image'], sample['mask'], sample['shadow']
                # Добавляем дополнительный канал для shadow (если shadow имеет 1 канал)
                img = np.dstack([img, shadow[..., 0]])
            return img / 255, mask
        else:
            return img / 255, mask


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
                # A.CLAHE(p=1),
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
        # ApplyToImageOnly(A.HueSaturationValue(p=0.9)),  # Применяем только к основному изображению
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
    return A.Compose(_transform, is_check_shapes=False, additional_targets={'shadow': 'image'})


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
                        channels=params.channels,
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
                 channels=['r', 'g', 'b'],
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
                            channels=channels,
                            target_size=target_size,
                            contour_thickness=contour_thickness,
                            max_workers=max_workers
                            )

    def dataset_test_fn(fp_data, aug):
        return CellDataset4Test(fp_data,
                                full_size=full_size,
                                add_shadow_to_img=add_shadow_to_img,
                                squares=squares,
                                border=border,
                                augmentation=aug,
                                preprocessing=preprocessing,
                                classes=classes,
                                channels=channels,
                                target_size=target_size,
                                contour_thickness=contour_thickness
                                )

    fp_data_list = SimpleNamespace(train=train_fp_data,
                                   valid=val_fp_data,
                                   test=test_fp_data)
    aug_list = SimpleNamespace(train=get_training_augmentation(target_size=transform_size),
                               valid=get_validation_augmentation(target_size=transform_size),
                               test=get_validation_augmentation(target_size=transform_size))

    return fp_data_list, aug_list, dataset_fn, dataset_test_fn
