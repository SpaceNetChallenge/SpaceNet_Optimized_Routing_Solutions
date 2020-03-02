
import time
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import os
import numpy as np
#import shutil
import logging
import json
import argparse

############
# need the following to avoid the following error:
#  TqdmSynchronisationWarning: Set changed size during iteration (see https://github.com/tqdm/tqdm/issues/481
from tqdm import tqdm
tqdm.monitor_interval = 0
############

from net.augmentations.transforms import get_flips_colors_augmentation, get_flips_shifts_augmentation
from net.dataset.abstract_image_provider import AbstractImageProvider
from net.dataset.reading_image_provider import ReadingImageProvider
from net.dataset.raw_image import RawImageType
from torch.utils.data.dataloader import DataLoader as PytorchDataLoader
from augmentations.composition import Compose
from augmentations.transforms import ToTensor
from net.dataset.image_cropper import ImageCropper
import random


class Dataset:
    """
    base class for datasets. for every image from image provider you will 
    have its own cropper
    """
    def __init__(self, image_provider: AbstractImageProvider, image_indexes, 
                 stage='train', transforms=None, verbose=True):
        self.pad = 0 if stage=='train' else 64
        self.image_provider = image_provider
        self.image_indexes = image_indexes if isinstance(image_indexes, list) \
                        else image_indexes.tolist()
        if verbose:
            print("nueral_dataset.py - Dataset - len imaage_indexes:", len(self.image_indexes))
        self.stage = stage
        self.keys = {'image', 'image_name'}
        self.transforms = Compose([transforms, ToTensor(10)])
        self.croppers = {}

    def __getitem__(self, item):
        raise NotImplementedError

    def get_cropper(self, image_id, val=False):
        #todo maybe cache croppers for different sizes too speedup if it's slow part?
        if image_id not in self.croppers:
            image = self.image_provider[image_id].image
            rows, cols = image.shape[:2]
            target_rows, target_cols = 512, 512
            cropper = ImageCropper(rows, cols,
                                   target_rows, target_cols,
                                   self.pad)
            self.croppers[image_id] = cropper
        return self.croppers[image_id]


class TrainDataset(Dataset):
    """
    dataset for training with random crops
    """
    def __init__(self, image_provider, image_indexes, stage='train', 
                 transforms=None, partly_sequential=False, verbose=True):
        super(TrainDataset, self).__init__(image_provider, image_indexes, 
             stage, transforms=transforms)
        self.keys.add('mask')
        self.partly_sequential = partly_sequential
        self.inner_idx = 9
        self.idx = 0
        if verbose:
            print("nueral_dataset.py - TrainDataset - len imaage_indexes:", len(image_indexes))

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

        #print ("neural_dataset.py data:", data)

        return self.transforms(**data)

    def __len__(self, verbose=False):
        z = len(self.image_indexes) * max(8, 1)
        return z # epoch size is len images


###############################################################################
if __name__ == "__main__":
    path_masks_train = '/wdata/train/masks_binned'
    path_images_train = '/wdata/train/psms' 

    paths = {
            'masks': path_masks_train,
            'images': path_images_train
            }
    
    fn_mapping = {
        'masks': lambda name: os.path.splitext(name)[0].replace('PS-MS', 'PS-RGB') + '.tif'  #'.png'
    }


    ds = ReadingImageProvider(RawImageType, paths, fn_mapping,
                              image_suffix='', num_channels=8)

    train_idx = np.arange(len(os.listdir(paths['images'])))
    transforms = get_flips_shifts_augmentation()

    train_loader = TrainDataset(ds, train_idx, transforms=transforms)

    b = 15
    gt = time.time()
    for m in range(2):
        st = time.time()
        for i, t in enumerate(train_loader):
            if i == b:
                break

    print('Time to iterate:', time.time() - gt, 'seconds')
