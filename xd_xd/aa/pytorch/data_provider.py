import os
import sys
from typing import Type, Dict, AnyStr, Callable

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import numpy as np
import skimage.io
import scipy.misc

from aa.cresi.utils import apls_tools


class AlphaNotAvailableException(Exception):
    pass


class AbstractImageType:
    """
    implement read_* methods in concrete image types. see raw_image for example
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha=False, num_channels=3):
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


class RawImageType(AbstractImageType):
    """
    image provider constructs image of type and then you can work with it
    """
    def __init__(self, paths, fn, fn_mapping, has_alpha, num_channels):
        super().__init__(paths, fn, fn_mapping, has_alpha)
        if num_channels == 3:
            self.im = skimage.io.imread(os.path.join(self.paths['images'], self.fn))
        else:
            self.im = apls_tools.load_multiband_im(os.path.join(self.paths['images'],
                                                                self.fn),
                                                   method='gdal')

    def read_image(self, verbose=False):
        if verbose:
            print("self:", self)
        im = self.im[...,:-1] if self.has_alpha else self.im
        if verbose:
            print("self.finalyze(im).shape", self.finalyze(im).shape)
        return self.finalyze(im)

    def read_mask(self, verbose=False):
        path = os.path.join(self.paths['masks'], self.fn_mapping['masks'](self.fn))
        # AVE edit:
        mask_channels = skimage.io.imread(path)
        # skimage reads in (channels, h, w) for multi-channel
        # assume less than 20 channels
        #print ("mask_channels.shape:", mask_channels.shape)
        if mask_channels.shape[0] < 20:
            #print ("mask_channels.shape:", mask_channels.shape)
            mask = np.moveaxis(mask_channels, 0, -1)
        else:
            mask = mask_channels

        ## original version (mode='L' is a grayscale black and white image)
        #mask = scipy.misc.imread(path, mode='L')
        if verbose:
            print ("raw_image.py mask.shape:", self.finalyze(mask).shape)
            print ("raw_image.py np.unique mask", np.unique(self.finalyze(mask)))
        return self.finalyze(mask)

    def read_alpha(self):
        return self.finalyze(self.im[...,-1])

    def finalyze(self, data):
        return self.reflect_border(data)


class AbstractImageProvider:
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


class ReadingImageProvider(AbstractImageProvider):
    def __init__(self,
                 image_type,
                 paths,
                 fn_mapping=lambda name: name,
                 image_suffix=None,
                 has_alpha=False,
                 num_channels=3):
        super(ReadingImageProvider, self).__init__(image_type,
                                                   fn_mapping,
                                                   has_alpha=has_alpha,
                                                   num_channels=num_channels)
        self.im_names = os.listdir(paths['images'])
        if image_suffix is not None:
            self.im_names = [n for n in self.im_names if image_suffix in n]

        self.paths = paths

    def get_indexes_by_names(self, names):
        return [idx for idx, name in enumerate(self.im_names)
                if os.path.splitext(name)[0] in names]

    def __getitem__(self, item):
        return self.image_type(self.paths,
                               self.im_names[item],
                               self.fn_mapping,
                               self.has_alpha,
                               self.num_channels)

    def __len__(self):
        return len(self.im_names)
