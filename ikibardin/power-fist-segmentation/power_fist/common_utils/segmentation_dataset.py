import numpy as np
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import random


class SegmentationBatchHandler:
    def __init__(self, training, num_tta):
        self._training = training
        self._num_tta = num_tta
        if self._training:
            assert self._num_tta is None

    def __call__(self, data):
        if self._training:
            return self._handle_batch_training(data)
        return self._handle_batch_predicting(data)

    @staticmethod
    def _handle_batch_training(data: Dict[str, torch.Tensor]) -> Tuple[Variable, Variable]:
        images = data['image']
        target = data['target']
        images = Variable(images.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        assert not torch.isnan(images).any()
        return images, target

    def _handle_batch_predicting(self, data):
        images = data['image']
        if self._num_tta is not None:
            _, _, c, h, w = images.size()
            images = images.view(-1, c, h, w)
        images = Variable(images.cuda(non_blocking=True))
        ids = data['id']
        return images, ids


class SegmentationDataset(Dataset):
    def __init__(self, paths_config, data_params, df, transform, mode):
        self._paths = paths_config
        self._df = df
        self._transform = transform
        assert mode in ('train', 'valid', 'test')
        self._mode = mode
        self._data_params = data_params

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, item):
        random.seed(random.randint(0, 666))
        np.random.seed(random.randint(0, 666))

        if isinstance(item, torch.Tensor):
            item = item.item()

        image_id = self._get_image_id(item)
        img = self._load_image(image_id)

        if self._mode == 'test':
            if self._transform is not None:
                img = self._transform(img)
            return {'image': img, 'id': image_id}

        mask, init_mask = self._load_mask(image_id)

        if self._transform is not None:
            img, mask = self._transform(img, mask)

        return {'image': img, 'mask': mask, 'id': image_id, 'init_mask': init_mask}

    def _get_image_id(self, item):
        raise NotImplementedError

    def _load_image(self, image_id):
        raise NotImplementedError

    def _load_mask(self, image_id):
        raise NotImplementedError
