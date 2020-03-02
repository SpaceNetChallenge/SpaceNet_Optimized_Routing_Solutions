import os
from typing import List

import numpy as np
import pandas as pd
import cv2
from scipy.stats import gmean

import torch

from power_fist.common_utils import transforms


class PredictionSaverInterface:
    def add(self, image_ids: List[str], predictions: np.ndarray):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def exists(self, image_id: str) -> bool:
        raise NotImplementedError


class SegmentationPredictionSaver(PredictionSaverInterface):
    def __init__(self, destination: str):
        super().__init__()
        self._dir = destination
        os.makedirs(self._dir, exist_ok=True)

        self._PNG_CHANNELS_COUNT = 3

    def add(self, image_ids: List[str], predictions: np.ndarray):
        predictions = self._prepare_predictions(predictions)
        # if len(image_ids) == 1:
        #     predictions = np.expand_dims(predictions, axis=0)
        for index, (id_, pred) in enumerate(zip(image_ids, predictions)):
            mask = np.zeros(shape=(pred.shape[0], pred.shape[1], pred.shape[2] + 1), dtype=np.uint8)
            mask[:, :, :-1] = (pred * 255.0).astype(np.uint8)

            this_sample_dir = os.path.join(self._dir, id_)
            os.makedirs(this_sample_dir, exist_ok=True)

            if mask.shape[2] % self._PNG_CHANNELS_COUNT != 0:
                additional_channels_count = (self._PNG_CHANNELS_COUNT - mask.shape[2] % self._PNG_CHANNELS_COUNT)
                additional_channels_mask = np.zeros(
                    shape=(mask.shape[0], mask.shape[1], additional_channels_count),
                    dtype=mask.dtype,
                )
                mask = np.dstack([mask, additional_channels_mask])

            for png_index, left_channel_index in enumerate(range(0, mask.shape[2], self._PNG_CHANNELS_COUNT)):
                cv2.imwrite(
                    os.path.join(this_sample_dir, f'{png_index}.png'),
                    mask[:, :, left_channel_index:left_channel_index + self._PNG_CHANNELS_COUNT, ],
                )

    def save(self):
        pass

    def _prepare_predictions(self, predictions: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def exists(self, image_id: str) -> bool:
        return os.path.exists(os.path.join(self._dir, image_id))


class SimplePredictionsSaver(SegmentationPredictionSaver):
    def __init__(self, destination):
        super().__init__(destination)

    def _prepare_predictions(self, predictions):
        return torch.sigmoid(predictions).data.cpu().numpy()


class _TTAAveragingSaver(SegmentationPredictionSaver):
    def __init__(self, destination, tta_size, size_before_padding):
        super().__init__(destination)
        self._tta_size = tta_size
        self.with_padding = size_before_padding is not None
        if self.with_padding:
            self._center_crop = transforms.OpenCVCenterCrop(size_before_padding)

    def _get_mean_masks(self, predictions):
        predictions = torch.sigmoid(predictions)
        _, c, h, w = predictions.size()
        predictions = predictions.data.view(-1, self._tta_size, c, h, w).cpu().numpy()
        predictions = predictions.transpose((0, 1, 3, 4, 2))
        masks = []
        for index in range(predictions.shape[0]):  # FIXME: slow
            mask = self._average_tta_pack(predictions[index])
            masks.append(mask)
        return masks

    def _prepare_predictions(self, predictions):
        masks = self._get_mean_masks(predictions)
        if self.with_padding:
            masks = [self._center_crop(mask) for mask in masks]
        return np.array(masks)

    def _average_tta_pack(self, tta_pack):
        raise NotImplementedError


class _TTAFullD4SaverBase(_TTAAveragingSaver):
    def __init__(self, destination, size_before_padding):
        self._FULL_D4_SIZE = 8
        super().__init__(destination, self._FULL_D4_SIZE, size_before_padding)

    def _average_tta_pack(self, d4_pack):
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
        return gmean(np.array(norm_orient), axis=0)

    @staticmethod
    def _get_rotated(img, angle):
        rows, cols = img.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows))


class TTAFullD4Saver(_TTAFullD4SaverBase):
    def __init__(self, destination):
        super().__init__(destination, size_before_padding=None)


class TTAFullD4WithPaddingSaver(_TTAFullD4SaverBase):
    def __init__(self, destination, initial_size):
        super().__init__(destination, size_before_padding=initial_size)


class _TTAHorizontalFlipSaverBase(_TTAAveragingSaver):
    def __init__(self, destination, size_before_padding):
        self._FULL_TTA_SIZE = 2
        super().__init__(destination, self._FULL_TTA_SIZE, size_before_padding)

    def _average_tta_pack(self, tta_pack):
        norm_orient = [tta_pack[0], cv2.flip(tta_pack[1], 1)]
        return gmean(np.array(norm_orient), axis=0)


class TTAHorizontalFlipSaver(_TTAHorizontalFlipSaverBase):
    def __init__(self, destination):
        super().__init__(destination, size_before_padding=None)


class TTAHorizontalFlipWithPaddingSaver(_TTAHorizontalFlipSaverBase):
    def __init__(self, destination, initial_size):
        super().__init__(destination, size_before_padding=initial_size)


class ClassificationTTAAveragingSaver(PredictionSaverInterface):
    def __init__(self, destination: str, tta_size: int, num_classes: int):
        super().__init__()
        self._dir = destination
        os.makedirs(self._dir, exist_ok=True)
        self._tta_size = tta_size
        self._num_classes = num_classes

        self._image_ids = []
        self._predicted_probas = None

    def add(self, image_ids: List[str], predictions: torch.Tensor):
        predictions = torch.sigmoid(predictions).data.cpu().numpy()
        predictions = predictions.reshape(len(image_ids), self._tta_size, self._num_classes)

        self._image_ids.extend(image_ids)

        predictions = np.mean(predictions, axis=1)  # FIXME gmean(predictions, axis=1)
        if self._predicted_probas is None:
            self._predicted_probas = predictions.copy()
        else:
            self._predicted_probas = np.append(self._predicted_probas, predictions.copy(), axis=0)

    def exists(self, image_id: str) -> bool:
        return False  # TODO: support partial predictions for classification

    def save(self):
        assert self._predicted_probas.shape[0] == len(
            self._image_ids
        ), (f'{self._predicted_probas.shape}'
            f' vs {len(self._image_ids)}')
        dataframe = pd.DataFrame(data=self._predicted_probas)
        dataframe['id'] = self._image_ids
        print(f'\nCreated predictions dataframe of shape {dataframe.shape}')

        output_path = os.path.join(self._dir, 'predictions.h5')
        dataframe.to_hdf(output_path, 'probs')
        print(f'Saved predictions dataframe to `{output_path}`\n')
