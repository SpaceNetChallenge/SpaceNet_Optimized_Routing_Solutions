__author__ = 'ikibardin'

import os
import multiprocessing
from typing import List, Dict

import numpy as np
import pandas as pd
import cv2
import skimage.io

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensor

from power_fist.common_utils import SegmentationDataInterface, saver, transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def post_transform() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensor(num_classes=1, sigmoid=False),
    ])


def hard_transform(crop_size: int, pad_height: int, pad_width: int) -> A.Compose:
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=(-0.2, 0.2),
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_REPLICATE,
            ),
            A.RandomCrop(crop_size, crop_size),
            A.Cutout(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            post_transform(),
        ]
    )


def hsv_transform(crop_size: int, pad_height: int, pad_width: int) -> A.Compose:
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=(-0.2, 0.2),
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_REPLICATE,
            ),
            A.RandomCrop(crop_size, crop_size),
            A.HueSaturationValue(p=0.5),
            A.Cutout(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            post_transform(),
        ]
    )


def hsv_no_cutout_transform(crop_size: int, pad_height: int, pad_width: int) -> A.Compose:
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=(-0.2, 0.2),
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_REPLICATE,
            ),
            A.RandomCrop(crop_size, crop_size),
            A.HueSaturationValue(p=0.5),
            # A.Cutout(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            post_transform(),
        ]
    )


def hsv_no_cutout_harder_transform(crop_size: int, pad_height: int, pad_width: int) -> A.Compose:
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.01,
                scale_limit=(-0.2, 0.2),
                rotate_limit=35,
                p=0.75,
                border_mode=cv2.BORDER_REPLICATE,
            ),
            A.RandomCrop(crop_size, crop_size),
            A.HueSaturationValue(p=0.5),
            # A.Cutout(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            post_transform(),
        ]
    )


def light_transform(crop_size: int, pad_height: int, pad_width: int) -> A.Compose:
    return A.Compose(
        [
            A.Cutout(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.PadIfNeeded(
                min_height=pad_height, min_width=pad_width, always_apply=True, p=1., border_mode=cv2.BORDER_REPLICATE,
            ),
            post_transform(),
        ]
    )


def valid_transform(crop_size: int, pad_height: int, pad_width: int) -> A.Compose:
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=pad_height, min_width=pad_width, always_apply=True, p=1., border_mode=cv2.BORDER_REPLICATE,
            ),
            post_transform(),
        ]
    )


def test_transform(**kwargs) -> A.Compose:
    return valid_transform(**kwargs)


def test_tta_d4(**kwargs) -> transforms._TTAFullD4Base:
    return transforms._TTAFullD4Base(final_transform=valid_transform(**kwargs))


AUGMENTATIONS = {
    'heavy': {
        'train': hard_transform,
        'valid': valid_transform,
        'test': test_transform,
    },

    'hsv': {
        'train': hsv_transform,
        'valid': valid_transform,
        'test': test_transform,
    },

    'hsv_no_cutout': {
        'train': hsv_no_cutout_transform,
        'valid': valid_transform,
        'test': test_transform,
    },

    'hsv_no_cutout_harder': {
        'train': hsv_no_cutout_harder_transform,
        'valid': valid_transform,
        'test': test_tta_d4,
    },

    'light': {
        'train': light_transform,
        'valid': valid_transform,
        'test': test_transform,
    },

}

MEDIAN_SPEED_IN_BINS = [5, 15, 25, 35, 45, 55, 65]


class ImageReader:
    def __init__(self, mode: str, paths_config: Dict[str, Dict[str, str]], activation: str):
        self._mode = mode
        assert mode in ('train', 'valid', 'test'), mode
        self._paths = paths_config
        assert activation in ('sigmoid', 'softmax'), activation
        self._activation = activation

    def load_image(self, image_id: str) -> np.ndarray:
        path = self._get_path(image_id, is_mask=False)
        img = cv2.imread(path)
        assert img is not None, path
        return img

    def load_mask(self, image_id: str) -> np.ndarray:
        mask_path = self._get_path(image_id, is_mask=True)
        mask = skimage.io.imread(mask_path)
        assert mask is not None, mask_path
        if self._activation == 'sigmoid':
            mask = mask[:, :, -1]
            # print('Mask max: ', mask.max())
            assert len(mask.shape) == 2, mask.shape
            return mask
        elif self._activation == 'softmax':
            h, w, _ = mask.shape
            probability_mask = np.zeros(shape=(len(MEDIAN_SPEED_IN_BINS), h, w), dtype=np.float32)
            for bin_index in range(len(MEDIAN_SPEED_IN_BINS)):
                bin_mask = mask[:, :, bin_index] > 127
                probability_mask[bin_index][bin_mask] = 0.6
                if bin_index > 0:
                    probability_mask[bin_index - 1][bin_mask] = 0.2
                else:
                    probability_mask[bin_index][bin_mask] += 0.2
                if bin_index < len(MEDIAN_SPEED_IN_BINS) - 1:
                    probability_mask[bin_index + 1][bin_mask] = 0.2
                else:
                    probability_mask[bin_index][bin_mask] += 0.2
            probability_mask[0][probability_mask.sum(axis=0) == 0.0] = 1.0
            # print(probability_mask[:, probability_mask.sum(axis=0) == 2.0])
            assert np.allclose(probability_mask.sum(axis=0), 1.0), \
                (probability_mask.sum(axis=0).min(), probability_mask.sum(axis=0).max())
            return probability_mask
        else:
            raise ValueError(f'Unknown activation: {self._activation}')

    def _get_path(self, image_id: str, is_mask: bool = False) -> str:
        dataset_paths = self._paths['dataset']
        if self._mode == 'test':
            assert not is_mask
            path = os.path.join(dataset_paths['test_dir'], f'{image_id}.tif')
            if not os.path.exists(path):
                train_path = os.path.join(dataset_paths['path'], dataset_paths['images_dir'], f'{image_id}.tif')
                if os.path.exists(train_path):
                    path = train_path
                else:
                    raise FileNotFoundError(
                        f'Image not found neither in test dir `{path}` nor in train dir `{train_path}`')
        elif is_mask:
            path = os.path.join(dataset_paths['path'], dataset_paths['masks_dir'], f'{image_id}.tif')
        else:
            path = os.path.join(dataset_paths['path'], dataset_paths['images_dir'], f'{image_id}.tif')

        assert os.path.exists(path), path
        return path


class SpacenetDataset(Dataset):
    def __init__(self, paths_config: dict, data_params: dict, df: pd.DataFrame, transform: A.Compose, mode: str):
        assert mode in ('train', 'valid', 'test'), mode
        self._paths = paths_config
        self._data_params = data_params
        self._df = df
        self._transform = transform
        self._mode = mode
        self.data_params = data_params

        self._image_reader = ImageReader(mode=mode, paths_config=paths_config, activation=data_params['activation'])
        self._activation = data_params['activation']

    def get_mode(self) -> str:
        return self._mode

    def __len__(self) -> int:
        return self._df.shape[0]

    def __getitem__(self, item: int) -> dict:
        image_id = self._get_image_id(item)
        result = {'id': image_id}

        image = self._image_reader.load_image(image_id)

        if self._mode == 'test':
            if self._transform is not None:
                image = self._transform(image=image)['image']
            result['image'] = image
            return result

        mask = self._image_reader.load_mask(image_id)

        if self._transform is not None:
            if self._activation == 'sigmoid':
                assert image.shape[:2] == mask.shape, (image.shape, mask.shape)
                tr = self._transform(image=image, mask=mask)
                image, mask = tr['image'], tr['mask']
            elif self._activation == 'softmax':
                assert image.shape[:2] == mask.shape[-2:], (image.shape, mask.shape)
                # print('BEFORE ALBU: ', mask.min(), mask.max(), mask.mean())
                tr = self._transform(image=image, masks=[mask[c] for c in range(mask.shape[0])])
                image, masks_list = tr['image'], tr['masks']
                mask = torch.stack(tuple(map(torch.FloatTensor, masks_list)))
                # print('AFTER  ALBU: ', mask.min(), mask.max(), mask.mean())
            else:
                raise ValueError(f'Unknown activation: {self._activation}')

        # print('>>>>>> ', mask.max())
        result['image'] = image
        result['target'] = mask
        return result

    def _get_image_id(self, item: int) -> str:
        if isinstance(item, torch.Tensor):
            item = item.item()
        return self._df.loc[item, 'id']


SAVER_POOL = None


class SpacenetPredictionsSaver(saver.PredictionSaverInterface):
    def __init__(self, destination: str, paths_config: Dict[str, Dict[str, str]], crop_height: int, crop_width: int,
                 activation: str):
        super().__init__()
        self._image_reader = ImageReader(mode='test', paths_config=paths_config, activation=activation)
        self._dir = destination
        assert activation in ('sigmoid', 'softmax'), activation
        self._activation = activation
        os.makedirs(self._dir, exist_ok=True)

        self._crop = A.CenterCrop(height=crop_height, width=crop_width, always_apply=True, p=1.)
        self._tta_size = 8

        global SAVER_POOL
        SAVER_POOL = multiprocessing.Pool(multiprocessing.cpu_count())

    def add(self, image_ids: List[str], predictions: torch.Tensor):
        predictions = self._prepare_predictions(predictions)
        results = []
        for id_, pred in zip(image_ids, predictions):
            results.append(SAVER_POOL.apply_async(self._add_single, (id_, pred)))
        for res in results:
            res.get()

    def _add_single(self, image_id: str, predictions: np.ndarray):
        # print('IN')
        if self._activation == 'sigmoid':
            mask = np.transpose((predictions * 255.0).astype(np.uint8))
            mask = self._crop(image=mask)['image']

            h, w, _ = self._image_reader.load_image(image_id=image_id).shape
            assert mask.shape == (h, w), mask.shape

            cv2.imwrite(
                os.path.join(self._dir, f'{image_id}.png'),
                mask,
            )
        elif self._activation == 'softmax':
            # mask = (predictions * 255.0).astype(np.uint8)
            mask = self._crop(image=predictions)['image']
            mask = np.transpose(mask, (2, 0, 1))
            h, w, _ = self._image_reader.load_image(image_id=image_id).shape
            assert mask.shape == (7, h, w), mask.shape

            speed_mask = self._get_speed_mask(mask)
            speed_mask = (speed_mask / 65.0 * 255.0).astype(np.uint8)

            cv2.imwrite(
                os.path.join(self._dir, f'{image_id}.png'),
                speed_mask,
            )
        else:
            raise ValueError(f'Unknown activation: {self._activation}')

    def save(self):
        pass

    def _prepare_predictions(self, predictions: torch.Tensor) -> np.ndarray:
        if self._activation == 'sigmoid':
            predictions = torch.sigmoid(predictions)
            _, c, h, w = predictions.size()
            predictions = predictions.data.view(-1, self._tta_size, c, h, w).cpu().numpy()
            predictions = predictions.transpose((0, 1, 3, 4, 2)).squeeze(-1)
            masks = []
            for index in range(predictions.shape[0]):  # FIXME: slow ?
                mask = self._average_tta_pack(predictions[index])
                # mask = cv2.flip(mask, flipCode=1)
                # mask = cv2.flip(mask, flipCode=0)
                mask = mask.T
                masks.append(mask)
            return np.array(masks)
        elif self._activation == 'softmax':
            predictions = torch.softmax(predictions, dim=1)
            _, c, h, w = predictions.size()
            predictions = predictions.data.view(-1, self._tta_size, c, h, w).cpu().numpy()
            predictions = predictions.transpose((0, 1, 3, 4, 2))
            masks = []
            for index in range(predictions.shape[0]):  # FIXME: slow ?
                mask = self._average_tta_pack(predictions[index])
                # mask = cv2.flip(mask, flipCode=1)
                # mask = cv2.flip(mask, flipCode=0)
                masks.append(mask)
            return np.array(masks)
        else:
            raise ValueError(f'Unknown activation: {self._activation}')

    def _average_tta_pack(self, d4_pack: np.ndarray) -> np.ndarray:
        # print('d4 pack ', d4_pack.shape)
        norm_orient = [
            d4_pack[0],
            self._get_rotated(d4_pack[1], 270),
            self._get_rotated(d4_pack[2], 180),
            self._get_rotated(d4_pack[3], 90),
            cv2.flip(d4_pack[4], 1),
            cv2.flip(self._get_rotated(d4_pack[5], 270), 1),
            cv2.flip(self._get_rotated(d4_pack[6], 180), 1),
            cv2.flip(self._get_rotated(d4_pack[7], 90), 1),
        ]
        return np.mean(np.array(norm_orient), axis=0)

    @staticmethod
    def _get_rotated(image: np.ndarray, angle: float) -> np.ndarray:
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (cols, rows))

    def exists(self, image_id: str) -> bool:
        output_path = os.path.join(self._dir, f'{image_id}.png')
        return os.path.exists(output_path)

    @staticmethod
    def _get_speed_mask(binned_mask: np.ndarray) -> np.ndarray:
        assert np.allclose(binned_mask.sum(axis=0), 1.0, rtol=0.1), \
            (binned_mask.sum(axis=0).min(), binned_mask.sum(axis=0).max())

        speed_mask = np.average(binned_mask, axis=0, weights=MEDIAN_SPEED_IN_BINS) * np.sum(MEDIAN_SPEED_IN_BINS)

        # print(' >>> ', speed_mask.shape, speed_mask.min(), speed_mask.mean(), speed_mask.max())
        assert len(speed_mask.shape) == 2, speed_mask.shape
        return speed_mask


class SpacenetDataInterface(SegmentationDataInterface):
    def __init__(self, paths_config: dict, data_params: dict, predict_params: dict):
        super().__init__(paths_config, data_params, predict_params, SpacenetDataset)

    def _make_transform(self, mode: str) -> A.Compose:
        return AUGMENTATIONS[self._data_params['augs']][mode](**self._data_params['augs_params'])

    def _make_train_transform(self) -> A.Compose:
        return self._make_transform(mode='train')

    def _make_valid_transform(self) -> A.Compose:
        return self._make_transform(mode='valid')

    def _make_test_transform(self):
        return self._make_transform(mode='test')

    def make_predictions_saver(
            self, experiment_name: str, checkpoint_path: str, dataset_part: str
    ) -> saver.PredictionSaverInterface:
        predictions_destination = self._get_predictions_destination(experiment_name, checkpoint_path, dataset_part)
        # print('Saving to', predictions_destination)
        return SpacenetPredictionsSaver(
            destination=predictions_destination,
            paths_config=self._paths_config,
            crop_height=1300,
            crop_width=1300,
            activation=self._data_params['activation'],
        )

    def _get_predictions_destination(self, experiment_name: str, checkpoint_path: str, dataset_part: str) -> str:
        stage_name = checkpoint_path.split('/')[-3]
        checkpoint_name = 'fold{}_{}_{}'.format(
            self._data_params['fold'], stage_name,
            checkpoint_path.split('/')[-1].split('.pth')[-2]
        )
        return os.path.join(
            self._paths_config['dumps']['path'],
            experiment_name,
            'fold_{}'.format(self._data_params['fold']),
            self._paths_config['dumps']['predictions'],
            checkpoint_name,
            dataset_part,
        )
