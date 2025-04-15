# This module defines dataset classes for the CellSeg project.
# It includes both precomputed datasets (CellDataset4) and test datasets (CellDataset4Test1, CellDataset4Test2, CellDataset4Test)
# with dynamic loading, caching, augmentation, and preprocessing functionality.

from pathlib import Path
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
        # Initialize dataset parameters and load precomputed images, masks, and auxiliary data
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

        # Process data either sequentially (default) or in parallel if max_workers > 0
        if self.max_workers <= 0:
            self.default_processing()
        else:
            self.parallel_processing(max_workers=self.max_workers)  # Launch parallel processing

        if len(self.images) != 0:
            self.channels_num = self.images[0].shape[-1]
            if self.add_shadow_to_img:
                self.channels_num += 1

    def process_fp_data(self, fp_data):
        """Processes a single data object.

        Reads the image and mask, prepares the mask (optionally with contours),
        splits the image into squares if required, and returns lists of images, masks, shadows, and info.
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
            # Split the image and mask into squares
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
        # Sequentially process each file pointer data and extend the dataset lists
        for fp_data in tqdm(self.all_fp_data):
            images, masks, shadows, info = self.process_fp_data(fp_data)
            self.images.extend(images)
            self.masks.extend(masks)
            self.shadows.extend(shadows)
            self.info.extend(info)

    def parallel_processing(self, max_workers=4):
        """Processes data in parallel using ThreadPoolExecutor."""
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
                    print(f"Error processing data: {e}")

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """Reads an image from file pointers for the specified channels."""
        if channels is None:
            channels = ['r', 'g', 'b']

        c_stack = list()
        for c_idx, c in enumerate(channels):
            c_img = Image.open(fp_data[f'{c}_fp'])

            # Optionally convert image if needed
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
                # Extract the channel using the index if no conversion is applied
                c_img = np.asarray(c_img)[..., c_idx]
            c_stack.append(c_img)

        img_np = np.stack(c_stack, axis=-1)

        return img_np

    @staticmethod
    def read_mask(fp_data):
        # Reads mask image using the 'mask' channel
        return CellDataset4.read_image(fp_data, channels=['mask'])

    @staticmethod
    def read_shadow(fp_data):
        # Reads shadow image using the 'p' channel and converts to grayscale
        return CellDataset4.read_image(fp_data, channels=['p'], convert='L')

    @staticmethod
    def _prepare_mask(mask, cls_num=1, cls=0, contour_thickness=2):
        """Prepares the mask: finds contours and creates an extra channel for the contour."""
        mask_contour = mask.copy()
        contours, _ = cv2.findContours(mask_contour,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, contourIdx=-1, color=[2], thickness=contour_thickness)
        mask = np.zeros((mask.shape[0], mask.shape[1], cls_num + 1), dtype=mask.dtype)
        mask[..., cls:cls + 1][mask_contour == 1] = 1
        mask[..., cls_num:cls_num + 1][mask_contour == 2] = 1

        return mask

    def aug(self, img, mask, shadow=False):
        """Applies augmentation and/or preprocessing if specified.
        If shadow is provided, it is added as an extra channel.
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

                # Combine the shadow channel with the image if provided (assumes shadow has one channel)
                img = np.dstack([img, shadow[..., 0]])

            return img / 255, mask
        else:
            return img / 255, mask

    def __getitem__(self, i):
        # Retrieve an augmented (or preprocessed) image and mask pair,
        # transposing them to (channels, height, width) format as float32 tensors.
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
        Parameters are similar to the original dataset but data is not precomputed.
        Building a mapping by indices allows dynamically loading the required square.
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

        # Build a mapping: for each global index, store a tuple (fp_data index, square index within the image)
        self.mapping = []
        for fp_idx, fp in enumerate(self.all_fp_data):
            if self.full_size is not None and self.squares is not None:
                # If splitting into squares is performed, add as many entries as there are squares
                for sq_idx, _ in enumerate(self.squares):
                    self.mapping.append((fp_idx, sq_idx))
            else:
                # If no splitting is needed, one entry per image
                self.mapping.append((fp_idx, None))

        # For speeding up sequential iteration: cache the last loaded file.
        self._cached_fp_index = None
        self._cached_data = None  # Will store a tuple (images, masks, shadows, info)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        """
        Returns the image and mask corresponding to the given index.
        Uses cache if the file is already loaded; otherwise, loads the data.
        After retrieving the last square for a file, clears the cache to release memory.
        """
        fp_idx, square_idx = self.mapping[index]

        # Use cache if available
        if self._cached_fp_index == fp_idx:
            images, masks, shadows, info = self._cached_data
        else:
            fp_data = self.all_fp_data[fp_idx]
            images, masks, shadows, info = self.process_fp_data(fp_data)
            self._cached_fp_index = fp_idx
            self._cached_data = (images, masks, shadows, info)

        # Select the specific square or full image if no splitting is applied.
        if square_idx is not None:
            img = images[square_idx]
            mask = masks[square_idx]
            shadow = shadows[square_idx]
        else:
            img = images[0]
            mask = masks[0]
            shadow = shadows[0]

        # Apply augmentation/preprocessing if specified
        img_aug, mask_aug = self.aug(img, mask, shadow=shadow)
        result = (img_aug.transpose(2, 0, 1).astype(np.float32),
                  mask_aug.transpose(2, 0, 1).astype(np.float32))

        # Clear cache if the current square is the last one for this file or it is the last element in the dataset
        if index == len(self.mapping) - 1 or self.mapping[index + 1][0] != fp_idx:
            self._cached_fp_index = None
            self._cached_data = None

        return result

    def process_fp_data(self, fp_data):
        """
        Loads the image and mask, applies initial processing,
        and splits them into squares if necessary.
        Returns lists of images, masks, shadows, and info (a dictionary with fp_data fields and square info).
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
            # Split the image and mask into squares.
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
                result_info.append(dict(sq=sq, **fp_data))
        else:
            result_images.append(img)
            result_masks.append(mask.astype(np.bool_))
            result_shadows.append(shadow)
            result_info.append(fp_data)

        return result_images, result_masks, result_shadows, result_info

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """
        Reads an image from file pointers for the specified channels.
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
        Processes the mask by finding contours and creating an extra channel for the contour.
        """
        mask_contour = mask.copy()
        contours, _ = cv2.findContours(mask_contour,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        _ = cv2.drawContours(mask_contour, contours, contourIdx=-1, color=[2], thickness=contour_thickness)
        mask_prepared = np.zeros((mask.shape[0], mask.shape[1], cls_num + 1),
                                 dtype=mask.dtype)
        mask_prepared[..., cls:cls + 1][mask_contour == 1] = 1
        mask_prepared[..., cls_num:cls_num + 1][mask_contour == 2] = 1
        return mask_prepared

    def aug(self, img, mask, shadow=False):
        """
        Applies augmentation and/or preprocessing if provided.
        If a shadow is provided, adds it as an extra channel.
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
                # Add an extra channel for shadow (assuming shadow has 1 channel)
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
        Similar to the original dataset but data is not precomputed.
        Builds a mapping from global index to (fp_data index, square index) to dynamically load the required square.
        Implements caching for multiple files and a mechanism to free memory once all squares for a file are requested.
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

        # Build mapping: for each global index, record (fp_data index, square index within the image)
        self.mapping = []
        # Also determine for each fp_data the total number of squares to be requested
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

        # Use a dictionary for caching to support multiple simultaneously loaded files.
        self._cache = {}         # key: fp_index, value: (images, masks, shadows, info)
        self._cache_usage = {}   # key: fp_index, value: number of squares already returned

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        """
        Returns the image and mask for the given index.
         - If the file data is cached, returns the cached data.
         - After the required number of squares for a file have been returned, the cache for that file is freed.
         - The corresponding info field is stored in self.info for later use in visualizations.
        """
        fp_idx, square_idx = self.mapping[index]

        # Load data from cache if available for this fp_idx.
        if fp_idx in self._cache:
            images, masks, shadows, info = self._cache[fp_idx]
        else:
            fp_data = self.all_fp_data[fp_idx]
            images, masks, shadows, info = self.process_fp_data(fp_data)
            self._cache[fp_idx] = (images, masks, shadows, info)
            self._cache_usage[fp_idx] = 0

        # If splitting is applied, choose the specified square; otherwise, use the full image.
        if square_idx is not None:
            img = images[square_idx]
            mask = masks[square_idx]
            shadow = shadows[square_idx]
        else:
            img = images[0]
            mask = masks[0]
            shadow = shadows[0]

        # Apply augmentation/preprocessing if specified.
        img_aug, mask_aug = self.aug(img, mask, shadow=shadow)
        result = (img_aug.transpose(2, 0, 1).astype(np.float32),
                  mask_aug.transpose(2, 0, 1).astype(np.float32))

        # Update info for the current index if needed.
        self.info[index] = info

        # Update cache usage count and free cache if all expected squares are served.
        self._cache_usage[fp_idx] += 1
        if self._cache_usage[fp_idx] >= self.file_total_counts[fp_idx]:
            del self._cache[fp_idx]
            del self._cache_usage[fp_idx]

        return result

    def process_fp_data(self, fp_data):
        """
        Loads the image and mask, performs initial processing, and splits them into squares if necessary.
        Returns lists of images, masks, shadows, and info (a dictionary with fp_data fields and 'sq' parameter).
        """
        img = self.read_image(fp_data, channels=self.channels)
        mask = self.read_mask(fp_data) // 255
        assert img.shape[:-1] == mask.shape[:-1]

        if self.classes is not None:
            mask = self._prepare_mask(mask, cls_num=len(self.classes), cls=self.classes.index(fp_data['cls']),
                                      contour_thickness=self.contour_thickness)
        else:
            mask = self._prepare_mask(mask, cls_num=1, cls=0,
                                      contour_thickness=self.contour_thickness)
        shadow = self.read_shadow(fp_data)[..., 0:1] if self.add_shadow_to_img else None

        result_images, result_masks, result_shadows, result_info = [], [], [], []

        if all(v is not None for v in [self.full_size, self.squares, self.border]):
            # Split the image and mask into squares.
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
                result_info.append(dict(sq=sq, **fp_data))

        else:
            result_images.append(img)
            result_masks.append(mask.astype(np.bool_))
            result_shadows.append(shadow)
            result_info.append(fp_data)

        return result_images, result_masks, result_shadows, result_info

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """
        Reads an image from file pointers for the specified channels.
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
        Extracts contours from the mask and creates an extra channel for the contour.
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
        Applies augmentation and/or preprocessing if provided.
        If shadow is available, adds it as an extra channel.
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
                # Add an extra channel for shadow (if shadow has 1 channel)
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
        Dataset for testing that dynamically loads and processes data.
        When using batch_size > 1, caching for multiple files is implemented with memory release once all squares of a file are requested.
        Additionally, info is saved for each element for later use in result visualization.
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

        # Build mapping: for each global index, record (fp_data index, square index)
        self.mapping = []
        for fp_idx, fp in enumerate(self.all_fp_data):
            if self.full_size is not None and self.squares is not None:
                for sq_idx in range(len(self.squares)):
                    self.mapping.append((fp_idx, sq_idx))
            else:
                self.mapping.append((fp_idx, None))

        # Create a list to store info for each element.
        self.info = [None] * len(self.mapping)

        # Build a dictionary for fast access: for each fp_index, list of indices in the mapping.
        self.mapping_indices = {}
        for idx, (fp_idx, square_idx) in enumerate(self.mapping):
            if fp_idx not in self.mapping_indices:
                self.mapping_indices[fp_idx] = []
            self.mapping_indices[fp_idx].append(idx)

        # Dictionary to cache file data. For each fp_index, store: (images, masks, shadows, info, expected_count)
        self._cache = {}
        # Cache usage counter for each fp_index.
        self._cache_usage = {}

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        """
        Returns the image and mask for the given index.
         - Uses cached data if available, otherwise loads the file.
         - Updates the info field for visualization.
         - Frees the cache once all squares of a file have been issued.
        """
        fp_idx, square_idx = self.mapping[index]

        # Check cache for the specified file index.
        if fp_idx in self._cache:
            images, masks, shadows, infos, expected_count = self._cache[fp_idx]
        else:
            fp_data = self.all_fp_data[fp_idx]
            images, masks, shadows, infos = self.process_fp_data(fp_data)
            expected_count = len(images)  # actual number of squares obtained
            self._cache[fp_idx] = (images, masks, shadows, infos, expected_count)
            self._cache_usage[fp_idx] = 0

            # Pre-fill all self.info entries corresponding to this fp_idx.
            for i in self.mapping_indices.get(fp_idx, []):
                _, sq = self.mapping[i]
                self.info[i] = infos[sq] if sq is not None else infos[0]

        # If splitting is applied, select the correct square; otherwise, take the first element.
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

        # Apply augmentation/preprocessing as needed.
        img_aug, mask_aug = self.aug(img, mask, shadow=shadow)
        result = (img_aug.transpose(2, 0, 1).astype(np.float32),
                  mask_aug.transpose(2, 0, 1).astype(np.float32))

        # Optionally update self.info for the current index.
        self.info[index] = info

        # Update cache usage for the given file.
        self._cache_usage[fp_idx] += 1
        # If all expected elements for this file are issued, free the cache.
        if self._cache_usage[fp_idx] >= expected_count:
            del self._cache[fp_idx]
            del self._cache_usage[fp_idx]

        return result

    def process_fp_data(self, fp_data):
        """
        Loads the image and mask, performs initial processing, and splits them into squares if required.
        Returns lists of images, masks, shadows, and infos (a dictionary containing fp_data fields and 'sq' information).
        """
        img = self.read_image(fp_data, channels=self.channels)
        mask = self.read_mask(fp_data)
        if mask is None:
            # if self.classes is not None:
            #     mask_dim = len(self.classes)
            # else:
            #     mask_dim = 1
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        else:
            mask = mask // 255

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
            # Split image and mask into squares.
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
                result_infos.append(dict(sq=sq, **fp_data))
        else:
            result_images.append(img)
            result_masks.append(mask.astype(np.bool_))
            result_shadows.append(shadow)
            result_infos.append(fp_data)

        return result_images, result_masks, result_shadows, result_infos

    @staticmethod
    def read_image(fp_data, channels=None, convert=None):
        """
        Reads an image from file pointers for the specified channels.
        """
        if channels is None:
            channels = ['r', 'g', 'b']
        c_stack = []
        for c_idx, c in enumerate(channels):
            if Path(fp_data[f'{c}_fp']).exists():
                c_img = Image.open(fp_data[f'{c}_fp'])
            else:
                return None
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
        Extracts contours from the mask and creates an extra channel for the contour.
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
        Applies augmentation and/or preprocessing if provided.
        If shadow is present, it is added as an extra channel.
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
                # Add additional channel for shadow (if shadow has 1 channel)
                img = np.dstack([img, shadow[..., 0]])
            return img / 255, mask
        else:
            return img / 255, mask


class ApplyToImageOnly(A.ImageOnlyTransform):
    def __init__(self, transform, always_apply=False, p=1.0):
        super(ApplyToImageOnly, self).__init__(always_apply, p)
        self.transform = transform

    def apply(self, img, **params):
        # Ensure that the transformation is applied only to the main image (3-channel)
        if img.shape[2] == 3:
            return self.transform(image=img)["image"]
        return img


def get_training_augmentation(target_size=None):
    # Define training augmentations with a probability of application
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
        # Uncomment the following line to apply HueSaturationValue only to the main image
        # ApplyToImageOnly(A.HueSaturationValue(p=0.9)),
    ]
    if target_size is not None:
        train_transform.append(A.Resize(height=target_size[1], width=target_size[0], always_apply=True))

    return A.Compose(train_transform, additional_targets={'shadow': 'image'})


def get_validation_augmentation(target_size=None):
    """Defines validation augmentations. Adds resizing if target size is specified."""
    test_transform = [
        # Uncomment and adjust the following lines if padding is needed:
        # A.PadIfNeeded(384, 480, always_apply=True, border_mode=0),
        # A.Resize(384, 480)
    ]
    if target_size is not None:
        test_transform.append(A.Resize(height=target_size[1], width=target_size[0], always_apply=True))
    return A.Compose(test_transform, is_check_shapes=False, additional_targets={'shadow': 'image'})


def to_tensor(x, **kwargs):
    return x.astype('float32')


def get_preprocessing(preprocessing_fn):
    """Constructs a preprocessing transform.

    Args:
        preprocessing_fn (callable): Data normalization function (can be specific for each pretrained network)

    Returns:
        Compose transform.
    """
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform, is_check_shapes=False, additional_targets={'shadow': 'image'})


def get_squares(orig_size, square_size, border):
    """Computes full image size, full size with borders, and square coordinates for splitting the image."""
    w, h = orig_size
    square_w, square_h = square_size

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

            square_coords = [start_w, start_h, end_w, end_h]
            square_with_borders_coords = [start_w, start_h, end_w + 2 * border, end_h + 2 * border]
            square_with_borders_coords_rev = [start_w + border, start_h + border, end_w + border, end_h + border]

            square_info_dict = dict(w=start_w_idx,
                                    h=start_h_idx,
                                    square_coords=square_coords,
                                    square_with_borders_coords=square_with_borders_coords,
                                    square_with_borders_coords_rev=square_with_borders_coords_rev)
            squares.append(square_info_dict)

    return full_size, full_size_with_borders, squares


def split_on_squares(image, squares, param_name):
    """Splits the image into a list of squares based on the provided square coordinates."""
    img_sq_list = []
    for sq in squares:
        img_sq = image[sq[param_name][1]:sq[param_name][3], sq[param_name][0]:sq[param_name][2], :].copy()
        img_sq_list.append(img_sq)

    return img_sq_list


def split_image(image, full_size, squares, border):
    """Resizes the image, adds borders, and splits it into squares."""
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
    """Reconstructs the full image from the split squares."""
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
    """Draws a colored square on the image."""
    for c_idx in range(len(color)):
        image[c_idx][sq[1]:sq[3], sq[0]] = color[c_idx]
        image[c_idx][sq[1], sq[0]:sq[2]] = color[c_idx]
        image[c_idx][sq[1]:sq[3], sq[2]] = color[c_idx]
        image[c_idx][sq[3], sq[0]:sq[2]] = color[c_idx]


def my_train_test_split(x, y, ratio_train, ratio_val, shuffle=True, seed=42):
    """Splits the data into training, validation, and test sets."""
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
    # Constructs data from provided parameters.
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
    # Retrieves file pointer data and splits it into training, validation, and test sets.
    all_fp_data = get_all_fp_data(dataset_dir, exp_class_dict)
    if shuffle:
        random.Random(42).shuffle(all_fp_data)
    all_fp_data = all_fp_data[:images_num]

    total_len = len(all_fp_data)
    train_num = int(total_len * ratio_train)
    val_num = int(total_len * ratio_val)

    train_fp_data = all_fp_data[:train_num]
    val_fp_data = all_fp_data[train_num:train_num + val_num]
    test_fp_data = all_fp_data[train_num + val_num:]

    img_path = all_fp_data[0]['p_fp']
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

    # preprocessing_fn can be specified for normalization (e.g., for specific encoders)
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
