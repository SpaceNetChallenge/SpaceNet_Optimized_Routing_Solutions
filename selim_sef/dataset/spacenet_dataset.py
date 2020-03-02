import os

import gdal
import numpy as np
import cv2
import pandas as pd
import skimage
import skimage.io
import torch
from albumentations.pytorch.transforms import img_to_tensor
from torch.utils.data import Dataset

cities = ['Vegas', 'Paris', 'Shanghai', 'Khartoum', 'Moscow', 'Mumbai']


def get_city_idx(file_name):
    for i, city in enumerate(cities):
        if city in file_name:
            return i
    raise ValueError("unknown city for {}".format(file_name))


class SpacenetSimpleDataset(Dataset):
    def __init__(self, data_path, mode, fold=0, folds_csv='folds.csv', transforms=None, normalize=None, multiplier=1,
                 image_type="PS-RGB"):
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.image_type=image_type
        self.names = sorted(os.listdir(os.path.join(self.data_path, image_type)))
        df = pd.read_csv(folds_csv)
        self.df = df
        self.normalize = normalize
        self.fold = fold
        if self.mode == "train":
            ids = set(df[df['fold'] != fold]['id'].tolist())
        else:
            ids = set(df[df['fold'] == fold]['id'].tolist())
        self.names = [n for n in self.names if n[:-4].replace("-MS", "-RGB") in ids]
        if mode == 'val':
            self.names = [n for n in self.names if 'Moscow' in n or 'Mumbai' in n or 'Vegas' in n]
        self.transforms = transforms
        if mode == "train":
            self.names = self.names * multiplier

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.data_path, self.image_type, name)
        image = skimage.io.imread(img_path)
        if image is None:
            raise ValueError(img_path)
        image_path = os.path.join(self.data_path, "train_mask_binned_mc_10", name[:-4].replace("RGB", "MS") + ".tif")
        try:
            raster_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
            speed_mask = raster_ds.ReadAsArray()
        except:
            # retry
            raster_ds = gdal.Open(image_path, gdal.GA_ReadOnly)
            speed_mask = raster_ds.ReadAsArray()
        if speed_mask.shape[-1] > 100:
            speed_mask = np.moveaxis(speed_mask, 0, -1)
        junction_mask = cv2.imread(os.path.join(self.data_path, "junction_masks", name[:-4].replace("PS-RGB_", "").replace("PS-MS_", "") + ".png"), cv2.IMREAD_GRAYSCALE)
        if junction_mask is None:
            junction_mask = np.zeros((1300, 1300))
        if speed_mask is None:
            speed_mask = np.zeros((1300, 1300, 11), dtype=np.uint8)
        sample = self.transforms(image=image, mask=np.concatenate([speed_mask, np.expand_dims(junction_mask, -1), np.expand_dims(junction_mask, -1)], axis=-1), img_name=name)
        mask = sample["mask"]
        city = np.zeros((len(cities), 1, 1))
        city[get_city_idx(name), 0, 0] = 1
        sample['city'] = city
        sample['img_name'] = name
        mask = np.moveaxis(mask, -1, 0) / 255.
        sample['mask'] = torch.from_numpy(mask)
        sample['image'] = img_to_tensor(sample["image"], self.normalize)
        return sample


class SpacenetSimpleDatasetTest(Dataset):
    def __init__(self, data_path, transforms=None, normalize=None):
        super().__init__()
        self.data_path = data_path
        self.names = [n for n in sorted(os.listdir(os.path.join(self.data_path))) if n.endswith(".tif")]
        self.transforms = transforms
        self.normalize=normalize
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img_path = os.path.join(self.data_path, name)
        image = skimage.io.imread(img_path)
        if image is None:
            raise ValueError(img_path)
        sample = self.transforms(image=image)
        sample['img_name'] = name
        sample['image'] = img_to_tensor(sample["image"], self.normalize)
        return sample
