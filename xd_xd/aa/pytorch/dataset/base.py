import os
from typing import Type, Dict, AnyStr, Callable

import numpy as np
from imageio import imread
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class AlphaNotAvailableException(Exception):
    pass


class AbstractImageType:
    """
    implement read_* methods in concrete image types.
    see raw_image for example
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha=False):
        self.paths = paths
        self.fn = fn
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping
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
            empty_x = np.zeros(
                (rows, cols, channels),
                dtype=image.dtype
            ) if channels else np.zeros((rows, cols), dtype=image.dtype)
            empty_x[0:image.shape[0], 0:image.shape[1], ...] = image
            image = empty_x
        return image

    def finalyze(self, image):
        return self.reflect_border(image)


class AbstractImageProvider:
    def __init__(self,
                 image_type: Type[AbstractImageType],
                 fn_mapping: Dict[AnyStr, Callable],
                 has_alpha=False):
        self.image_type = image_type
        self.has_alpha = has_alpha
        self.fn_mapping = fn_mapping

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ReadingImageProvider(AbstractImageProvider):
    def __init__(self, image_type, paths,
                 fn_mapping=lambda name: name,
                 image_suffix=None,
                 has_alpha=False):
        super(ReadingImageProvider, self).__init__(
            image_type, fn_mapping, has_alpha=has_alpha)
        self.im_names = os.listdir(paths['images'])
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names
                             if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        return [idx for idx, name in enumerate(self.im_names)
                if os.path.splitext(name)[0] in names]

    def __getitem__(self, item):
        return self.image_type(self.paths,
                               self.im_names[item],
                               self.fn_mapping,
                               self.has_alpha)

    def __len__(self):
        return len(self.im_names)


class ReadingImageOverAOIAugProvider(AbstractImageProvider):
    def __init__(self, image_type, paths, suffix_list, im_names,
                 fn_mapping=lambda name: name,
                 image_suffix=None,
                 has_alpha=False):
        super(ReadingImageOverAOIProvider, self).__init__(
            image_type, fn_mapping, has_alpha=has_alpha)
        self.im_names = im_names
        self.suffix_list = suffix_list

        if image_suffix is not None:
            self.im_names = [n for n in self.im_names
                             if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        return [idx for idx, name in enumerate(self.im_names)
                if os.path.splitext(name)[0] in names]

    def __getitem__(self, item):
        return self.image_type(self.paths,
                               self.im_names[item],
                               self.suffix_list[item],
                               self.fn_mapping,
                               self.has_alpha)

    def __len__(self):
        return len(self.im_names)


class ReadingImageOverAOIProvider(AbstractImageProvider):
    def __init__(self, image_type, paths, im_names,
                 fn_mapping=lambda name: name,
                 image_suffix=None,
                 has_alpha=False):
        super(ReadingImageOverAOIProvider, self).__init__(
            image_type, fn_mapping, has_alpha=has_alpha)
        self.im_names = im_names

        if image_suffix is not None:
            self.im_names = [n for n in self.im_names
                             if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        return [idx for idx, name in enumerate(self.im_names)
                if os.path.splitext(name)[0] in names]

    def __getitem__(self, item):
        return self.image_type(self.paths,
                               self.im_names[item],
                               self.fn_mapping,
                               self.has_alpha)

    def __len__(self):
        return len(self.im_names)


class ReadingImageListProvider(AbstractImageProvider):
    def __init__(self, image_type, paths,
                 filename_list,
                 fn_mapping=lambda name: name,
                 image_suffix=None,
                 has_alpha=False):
        super(ReadingImageListProvider, self).__init__(
            image_type, fn_mapping, has_alpha=has_alpha)
        self.im_names = filename_list
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names
                             if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        return [idx for idx, name in enumerate(self.im_names)
                if os.path.splitext(name)[0] in names]

    def __getitem__(self, item):
        return self.image_type(self.paths,
                               self.im_names[item],
                               self.fn_mapping,
                               self.has_alpha)

    def __len__(self):
        return len(self.im_names)


class RawImageType(AbstractImageType):
    """
    image provider constructs image of type and then
    you can work with it
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        # self.im = imread(os.path.join(self.paths['images'], self.fn),
        #                  pilmode='RGB')
        self.im = imread(os.path.join(self.paths['images'], self.fn))

    def read_image(self):
        im = self.im[..., :-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self):
        path = os.path.join(self.paths['masks'],
                            self.fn_mapping['masks'](self.fn))
        mask = imread(path, pilmode='L')
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[..., -1])

    def finalyze(self, data):
        return self.reflect_border(data)


class RawImageTypeMTL(AbstractImageType):
    """
    image provider constructs image of type and then
    you can work with it
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        filepath = os.path.join(self.paths['images'], self.fn + '.png')
        self.im = imread(filepath)

    def read_image(self):
        im = self.im[..., :-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self):
        path = os.path.join(self.paths['masks'], self.fn + '.png')
        mask = imread(path, pilmode='L')
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[..., -1])

    def finalyze(self, data):
        return self.reflect_border(data)


class RawImageTypeOverAOI(AbstractImageType):
    """
    image provider constructs image of type and then
    you can work with it
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        # self.im = imread(os.path.join(self.paths['images'], self.fn),
        #                  pilmode='RGB')
        # print(self.paths['images'], self.fn)
        filepath = os.path.join(self.paths['images'], self.fn + '.png')
        self.im = imread(filepath)

    def read_image(self):
        im = self.im[..., :-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self):
        # print(self.paths['masks'], self.fn)
        path = os.path.join(self.paths['masks'], self.fn + '.png')
        mask = imread(path, pilmode='L')
        # if config.num_classes > 1:
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[..., -1])

    def finalyze(self, data):
        return self.reflect_border(data)


class RawImageTypeOverAOIAug(AbstractImageType):
    """
    image provider constructs image of type and then
    you can work with it
    """
    def __init__(self, paths, fn, suffix, fn_mapping, has_alpha):
        super().__init__(paths, fn, suffix, fn_mapping, has_alpha)
        filepath = os.path.join(self.paths['images'] + suffix,
                                self.fn + '.png')
        self.im = imread(filepath)
        self.suffix = suffix

    def read_image(self):
        im = self.im[..., :-1] if self.has_alpha else self.im
        return self.finalyze(im)

    def read_mask(self):
        # print(self.paths['masks'], self.fn)
        path = os.path.join(self.paths['masks'] + self.suffix,
                            self.fn + '.png')
        mask = imread(path, pilmode='L')
        # if config.num_classes > 1:
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[..., -1])

    def finalyze(self, data):
        return self.reflect_border(data)


class RawImageTypePad(RawImageType):
    def finalyze(self, data):
        return self.reflect_border(data, 22)
