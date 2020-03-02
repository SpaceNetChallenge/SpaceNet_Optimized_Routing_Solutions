import random
from typing import Tuple

import numpy as np
import cv2

import torch
from torchvision import transforms

import albumentations
from albumentations import DualTransform
from albumentations.augmentations import functional as F
from albumentations import (
    CenterCrop,
    CLAHE,
    RandomRotate90,
    HorizontalFlip,
    Transpose,
    ShiftScaleRotate,
    Blur,
    OpticalDistortion,
    GridDistortion,
    HueSaturationValue,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    MotionBlur,
    MedianBlur,
    IAAPiecewiseAffine,
    IAASharpen,
    IAAEmboss,
    RandomBrightnessContrast,
    Flip,
    OneOf,
    Compose,
    PadIfNeeded,
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _strong_aug(crop_size, p):
    return Compose(
        [
            # RandomCrop(crop_size, crop_size),
            PadIfNeeded(crop_size, crop_size),
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([IAAAdditiveGaussianNoise(), GaussNoise()], p=0.2),
            OneOf(
                [
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    Blur(blur_limit=3, p=0.1),
                ],
                p=0.2,
            ),
            ShiftScaleRotate(shift_limit=0.0625 * 2, scale_limit=0.3, rotate_limit=45, p=0.75),
            OneOf(
                [
                    OpticalDistortion(p=0.3),
                    GridDistortion(p=0.1),
                    IAAPiecewiseAffine(p=0.3),
                ],
                p=0.2,
            ),
            OneOf(
                [
                    CLAHE(clip_limit=2),
                    IAASharpen(),
                    IAAEmboss(),
                    RandomBrightnessContrast,
                ],
                p=0.3,
            ),
            HueSaturationValue(p=0.15),
        ],
        p=p,
    )


def lighter_aug(pad_size, p):
    return Compose(
        [
            PadIfNeeded(pad_size, pad_size),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0, rotate_limit=5, p=0.75),
            RandomBrightnessContrast(p=0.5),
        ],
        p=p,
    )


class ToTensorAndNormalizeImageOnly:
    def __init__(self):
        self.func = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image):
        return self.func(image)


class ToTensorAndNormalizeWithMask:
    def __init__(self):
        self.totensor_normalize = ToTensorAndNormalizeImageOnly()

    def __call__(self, image, mask=None):
        image = self.totensor_normalize(image)
        if mask is not None:
            mask = torch.FloatTensor(mask)
        return image, mask


class PadAndNormalize:
    def __init__(self, pad_size=0):
        self.pad = PadIfNeeded(min_height=pad_size, min_width=pad_size)
        self.norm = ToTensorAndNormalizeWithMask()

    def __call__(self, image, mask=None):
        if mask is None:
            return self.norm(self.pad(image=image)['image'], mask)
        padded_data = self.pad(image=image, mask=mask)
        return self.norm(padded_data['image'], padded_data['mask'])


class PadAndNormalizeImageOnly:
    def __init__(self, pad_size):
        self.pad = PadAndNormalize(pad_size)

    def __call__(self, image):
        image, _ = self.pad(image, mask=None)
        return image


class HeavyTrainTransform:
    def __init__(self, crop_size, lighter=False):
        self.strong = (_strong_aug(crop_size, p=1.0) if not lighter else lighter_aug(crop_size, p=1.0))
        self.common = ToTensorAndNormalizeWithMask()

    def __call__(self, image, mask):
        augmented_data = self.strong(image=image, mask=mask)
        return self.common(augmented_data['image'], augmented_data['mask'])


class ValidationTransform:
    def __init__(self, crop_size):
        self.center_crop = CenterCrop(crop_size, crop_size, p=1.0)
        self.common = ToTensorAndNormalizeWithMask()

    def __call__(self, image, mask):
        cropped_data = self.center_crop(image=image, mask=mask)
        return self.common(cropped_data['image'], cropped_data['mask'])


####################################################################################


class OpenCVCropBase(object):
    def __init__(self, size):
        self._size = size

    def __call__(self, image):
        if image.shape[0] < self._size or image.shape[1] < self._size:
            raise ValueError(
                'Image too small: crop_size={}, while got'
                'image with shape {}'.format(self._size, image.shape)
            )
        i, j = self._get_params(image)
        assert 0 <= i <= image.shape[0]
        assert 0 <= j <= image.shape[1]
        assert 0 <= i + self._size <= image.shape[0]
        assert 0 <= j + self._size <= image.shape[1]
        return image[i:i + self._size, j:j + self._size]

    def _get_params(self, image):
        raise NotImplementedError


class OpenCVCenterCrop(OpenCVCropBase):
    def __init__(self, size):
        super().__init__(size)

    def _get_params(self, image):
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        if h == self._size and w == self._size:
            return 0, 0
        th = tw = self._size
        i = int(round((h - th) / 2.0))
        j = int(round((w - tw) / 2.0))
        return i, j


class FiveCrop:
    def __init__(self, crop_size: int):
        self._crop_size = crop_size
        self._center_crop = OpenCVCenterCrop(self._crop_size)

    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, w, c = image.shape
        assert c == 3, 'Something wrong with channels order'
        if self._crop_size > w or self._crop_size > h:
            raise ValueError('Requested crop size {} is greater than input size {}'.format(self._crop_size, (h, w)))
        tl = image[0:self._crop_size, 0:self._crop_size]
        tr = image[0:self._crop_size, w - self._crop_size:w]
        bl = image[h - self._crop_size:h, 0:self._crop_size]
        br = image[h - self._crop_size:h, w - self._crop_size:w]
        return tl, tr, bl, br, self._center_crop(image)


class _TTAFullD4Base:
    def __init__(self, final_transform):
        self._final = final_transform

    def __call__(self, image):
        # import pdb; pdb.set_trace();
        # y = self._get_all_90_rotations(image)[0]

        # print(type(y))
        initial_rotations = [self._final(image=x)['image'] for x in self._get_all_90_rotations(image)]
        flipped_rotations = [self._final(image=x)['image'] for x in
                             self._get_all_90_rotations(cv2.flip(image.copy(), 1))]
        return {'image': torch.stack(initial_rotations + flipped_rotations)}

    @staticmethod
    def _get_all_90_rotations(image):
        result = [image.copy()]
        for angle in [90, 180, 270]:
            result.append(_TTAFullD4Base._get_rotated(image.copy(), angle))
        return result

    @staticmethod
    def _get_rotated(image, angle):
        rows, cols, c = image.shape
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (cols, rows))


class TTAFullD4(_TTAFullD4Base):
    def __init__(self):
        super().__init__(ToTensorAndNormalizeImageOnly())


class TTAFullD4WithPadding(_TTAFullD4Base):
    def __init__(self, pad_size):
        super().__init__(PadAndNormalizeImageOnly(pad_size))


class TTAHorizontalFlipsWithPadding:
    def __init__(self, pad_size):
        self._final = PadAndNormalizeImageOnly(pad_size)

    def __call__(self, image):
        return torch.stack([self._final(image.copy()), self._final(cv2.flip(image.copy(), 1))])


class TTAHorizontalFlips:
    def __init__(self):
        self._final = ToTensorAndNormalizeImageOnly()

    def __call__(self, image):
        return torch.stack([self._final(image.copy()), self._final(cv2.flip(image.copy(), 1))])


class MakeSquare(albumentations.core.transforms_interface.BasicTransform):
    def __init__(self, img_size=224, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.img_size = img_size

    def apply(self, img, **params):
        GRAY = [127, 127, 127]
        if img.shape[0] < img.shape[1]:
            x1 = (img.shape[1] - img.shape[0]) // 2
            x2 = img.shape[1] - img.shape[0] - x1
            img = cv2.copyMakeBorder(img, x1, x2, 0, 0, cv2.BORDER_CONSTANT, value=GRAY)

        else:
            x1 = (img.shape[0] - img.shape[1]) // 2
            x2 = img.shape[0] - img.shape[1] - x1
            img = cv2.copyMakeBorder(img, 0, 0, x1, x2, cv2.BORDER_CONSTANT, value=GRAY)

        img = cv2.resize(img, (self.img_size, self.img_size))
        return img

    @property
    def targets(self):
        return {'image': self.apply}


class PowerFistRandomSizedCrop(DualTransform):
    """Crop a random part of the input and rescale it to some size.

    Args:
        min_max_height ((int, int)): crop size limits.
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        w2h_ratio (float): aspect ratio of crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            min_max_height,
            height,
            width,
            w2h_ratio=(3. / 4., 4. / 3.),
            interpolation=cv2.INTER_LINEAR,
            always_apply=False,
            p=1.0
    ):
        super(PowerFistRandomSizedCrop, self).__init__(always_apply, p)
        self.height = height
        self.width = width
        self.interpolation = interpolation
        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        h, w, _ = img.shape
        crop = F.random_crop(img, min(h, crop_height), min(w, crop_width), h_start, w_start)
        return F.resize(crop, self.height, self.width, interpolation)

    def get_params(self):
        crop_height = random.randint(self.min_max_height[0], self.min_max_height[1])
        w2h_ratio = random.uniform(self.w2h_ratio[0], self.w2h_ratio[1])
        return {
            'h_start': random.random(),
            'w_start': random.random(),
            'crop_height': crop_height,
            'crop_width': int(crop_height * w2h_ratio)
        }

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)

    def apply_to_keypoint(self, keypoint, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        keypoint = F.keypoint_random_crop(keypoint, crop_height, crop_width, h_start, w_start, rows, cols)
        scale_x = self.width / crop_height
        scale_y = self.height / crop_height
        keypoint = F.keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint


class CenterCropIfNeeded(albumentations.CenterCrop):
    def __init__(self, height, width, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        h, w, _ = img.shape
        return F.center_crop(img, min(self.height, h), min(self.width, w))
