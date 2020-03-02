import os
import sys
import random
from typing import Type, Dict, AnyStr, Callable

import numpy as np
import matplotlib.pyplot as plt
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# import relative paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from aa.cresi.net.augmentations.composition import Compose
from aa.cresi.net.augmentations.transforms import ToTensor


class AlphaNotAvailableException(Exception):
    pass


class AbstractImageType(object):
    """
    implement read_* methods in concrete image types. see raw_image for example
    """
    def __init__(self, paths, fn, fn_mapping,
                 has_alpha=False,
                 num_channels=3):
        self.paths = paths
        self.fn = fn
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping
        self.num_channels=num_channels
        self.cache = {}

    @property
    def image(self):
        if 'image' not in self.cache:
            self.cache['image'] = self.read_image()
        return self.cache['image']

    @property
    def mask(self):
        if 'mask' not in self.cache:
            self.cache['mask'] = self.read_mask()
        return self.cache['mask']

    @property
    def alpha(self):
        if not self.has_alpha:
            raise AlphaNotAvailableException
        if 'alpha' not in self.cache:
            self.cache['alpha'] = self.read_alpha()
        return self.cache['alpha']

    def read_alpha(self):
        raise NotImplementedError

    def read_image(self):
        raise NotImplementedError

    def read_mask(self):
        raise NotImplementedError

    def reflect_border(self, image, b=12):
        return cv2.copyMakeBorder(image, b, b, b, b, cv2.BORDER_REFLECT)

    def pad_image(self, image, rows, cols):
        channels = image.shape[2] if len(image.shape) > 2 else None
        if image.shape[:2] != (rows, cols):
            empty_x = np.zeros((rows, cols, channels), dtype=image.dtype) \
                    if channels else np.zeros((rows, cols), dtype=image.dtype)
            empty_x[0:image.shape[0],0:image.shape[1],...] = image
            image = empty_x
        return image

    def finalyze(self, image):
        return self.reflect_border(image)


class AbstractImageProvider(object):
    def __init__(self, image_type: Type[AbstractImageType],
                 fn_mapping: Dict[AnyStr, Callable], has_alpha=False,
                 num_channels=3):
        self.image_type = image_type
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping
        self.num_channels = num_channels

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ImageCropper(object):
    """
    generates random or sequential crops of image
    """
    def __init__(self, img_rows, img_cols, target_rows, target_cols, pad):
        self.image_rows = img_rows
        self.image_cols = img_cols
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.pad = pad
        self.use_crop = (img_rows != target_rows) or (img_cols != target_cols)
        self.starts_y = self.sequential_starts(axis=0) if self.use_crop else [0]
        self.starts_x = self.sequential_starts(axis=1) if self.use_crop else [0]
        self.positions = [(x, y) for x in self.starts_x for y in self.starts_y]
        # self.lock = threading.Lock()

    def random_crop_coords(self, verbose=False):
        x = random.randint(0, self.image_cols - self.target_cols)
        y = random.randint(0, self.image_rows - self.target_rows)
        return x, y

    def crop_image(self, image, x, y, verbose=False):
        z = image[y: y+self.target_rows, x: x+self.target_cols,...] if self.use_crop else image
        return z


    def sequential_crops(self, img):
        for startx in self.starts_x:
            for starty in self.starts_y:
                yield self.crop_image(img, startx, starty)

    def sequential_starts(self, axis=0, verbose=False):
        """
        splits range uniformly to generate uniform image crops with minimal pad (intersection)
        """
        big_segment = self.image_cols if axis else self.image_rows
        small_segment = self.target_cols if axis else self.target_rows
        if big_segment == small_segment:
            return [0]
        steps = np.ceil((big_segment - self.pad) / (small_segment - self.pad)) # how many small segments in big segment
        if steps == 1:
            return [0]
        new_pad = int(np.floor((small_segment * steps - big_segment) / (steps - 1))) # recalculate pad
        starts = [i for i in range(0, big_segment - small_segment, small_segment - new_pad)]
        starts.append(big_segment - small_segment)
        return starts


class Dataset(object):
    """
    base class for datasets. for every image from image provider you will
    have its own cropper
    """
    def __init__(self,
                 image_provider: AbstractImageProvider,
                 image_indexes,
                 config,
                 stage='train',
                 transforms=None,
                 verbose=True):
        self.pad = 0 if stage=='train' else config.test_pad
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) \
                        else image_indexes.tolist()
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.config = config
        self.transforms = Compose([transforms, ToTensor(config.num_classes)])
        self.croppers = {}

    def __getitem__(self, item):
        raise NotImplementedError

    def get_cropper(self, image_id, val=False):
        #todo maybe cache croppers for different sizes too speedup if it's slow part?
        if image_id not in self.croppers:
            image = self.image_provider[image_id].image
            rows, cols = image.shape[:2]
            if self.config.ignore_target_size and val:
                # we can igore target size if we want to validate on full images
                assert self.config.predict_batch_size == 1
                target_rows, target_cols = rows, cols
            else:
                target_rows, target_cols = self.config.target_rows, self.config.target_cols
            cropper = ImageCropper(rows, cols,
                                   target_rows, target_cols,
                                   self.pad)
            self.croppers[image_id] = cropper
        return self.croppers[image_id]


class TrainDataset(Dataset):
    """
    dataset for training with random crops
    """
    def __init__(self, image_provider, image_indexes, config,
                 stage='train',
                 transforms=None,
                 partly_sequential=False,
                 verbose=True):
        super(TrainDataset, self).__init__(
            image_provider, image_indexes,
             config, stage,
            transforms=transforms)

        self.keys.add('mask')
        self.partly_sequential = partly_sequential
        self.inner_idx = 9
        self.idx = 0

    def __getitem__(self, idx, verbose=False):
        if self.partly_sequential:
            # use this if your images are too big
            if self.inner_idx > 8:
                self.idx = idx
                self.inner_idx = 0
            self.inner_idx += 1
            im_idx = self.image_indexes[self.idx % len(self.image_indexes)]
        else:
            im_idx = self.image_indexes[idx % len(self.image_indexes)]

        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        sx, sy = cropper.random_crop_coords()
        if cropper.use_crop and self.image_provider.has_alpha:
            for i in range(10):
                alpha = cropper.crop_image(item.alpha, sx, sy)
                if np.mean(alpha) > 5:
                    break
                sx, sy = cropper.random_crop_coords()
            else:
                return self.__getitem__(random.randint(0, len(self.image_indexes)))

        im = cropper.crop_image(item.image, sx, sy)
        if not np.any(im > 5):
            # re-try random if image is empty
            return self.__getitem__(random.randint(0, len(self.image_indexes)))
        mask = cropper.crop_image(item.mask, sx, sy)
        data = {'image': im, 'mask': mask, 'image_name': item.fn}

        return self.transforms(**data)

    def __len__(self, verbose=False):
        z = len(self.image_indexes) * max(self.config.epoch_size, 1)
        return z # epoch size is len images

class SequentialDataset(Dataset):
    """
    dataset for test and base dataset for validation.
    produces sequential crops of images
    """
    def __init__(self, image_provider, image_indexes, config,
                 stage='test',
                 transforms=None):
        super(SequentialDataset, self).__init__(image_provider,
                                                image_indexes,
                                                config,
                                                stage,
                                                transforms=transforms)
        self.good_tiles = []
        self.init_good_tiles()
        self.keys.update({'geometry'})

    def init_good_tiles(self, verbose=False):
        self.good_tiles = []
        for im_idx in self.image_indexes:
            cropper = self.get_cropper(im_idx, val=True)
            positions = cropper.positions
            if self.image_provider.has_alpha:
                item = self.image_provider[im_idx]
                alpha_generator = cropper.sequential_crops(item.alpha)
                for idx, alpha in enumerate(alpha_generator):
                    if np.mean(alpha) > 5:
                        self.good_tiles.append((im_idx, *positions[idx]))
            else:
                for pos in positions:
                    self.good_tiles.append((im_idx, *pos))

    def prepare_image(self, item, cropper, sx, sy, verbose=False):
        im = cropper.crop_image(item.image, sx, sy)
        rows, cols = item.image.shape[:2]
        geometry = {'rows': rows, 'cols': cols, 'sx': sx, 'sy': sy}
        data = {'image': im, 'image_name': item.fn, 'geometry': geometry}
        return data

    def __getitem__(self, idx, verbose=False):
        if idx >= self.__len__():
            return None
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)
        return self.transforms(**data)

    def __len__(self):
        return len(self.good_tiles)


class ValDataset(SequentialDataset):
    """
    same as sequential but added mask
    """
    def __init__(self,
                 image_provider,
                 image_indexes,
                 config,
                 stage='train',
                 transforms=None):
        super(ValDataset, self).__init__(image_provider,
                                         image_indexes,
                                         config,
                                         stage,
                                         transforms=transforms)
        self.keys.add('mask')

    def __getitem__(self, idx, verbose=False):
        im_idx, sx, sy = self.good_tiles[idx]
        cropper = self.get_cropper(im_idx)
        item = self.image_provider[im_idx]
        data = self.prepare_image(item, cropper, sx, sy)
        mask = cropper.crop_image(item.mask, sx, sy)
        data.update({'mask': mask})
        return self.transforms(**data)
