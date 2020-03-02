import random
import numpy as np
import math
from functools import wraps

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import aa.cresi.net.augmentations.functional as F


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


def clipped(func):
    """
    wrapper to clip results of transform to image dtype value range
    """
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)
    return wrapped_function


def fix_shift_values(img, *args):
    """
    shift values are normally specified in uint, but if your data is float - you need to remap values
    """
    if img.dtype == np.float32:
        return list(map(lambda x: x / 255, args))
    return args


def vflip(img):
    return cv2.flip(img, 0)


def hflip(img):
    return cv2.flip(img, 1)


def flip(img, code):
    return cv2.flip(img, code)


def transpose(img):
    return img.transpose(1, 0, 2) if len(img.shape) > 2 else img.transpose(1, 0)


def rot90(img, times):
    img = np.rot90(img, times)
    return np.ascontiguousarray(img)


def rotate(img, angle):
    """
    rotate image on specified angle
    :param angle: angle in degrees
    """
    height, width = img.shape[0:2]
    mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
    img = cv2.warpAffine(img, mat, (width, height),
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return img


def shift_scale_rotate(img, angle, scale, dx, dy):
    """
    :param angle: in degrees
    :param scale: relative scale
    """
    height, width = img.shape[:2]

    cc = math.cos(angle/180*math.pi) * scale
    ss = math.sin(angle/180*math.pi) * scale
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0],  [width, height], [0, height], ])
    box1 = box0 - np.array([width/2, height/2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width/2+dx*width, height/2+dy*height])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    img = cv2.warpPerspective(img, mat, (width, height),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    return img


def center_crop(img, height, width):
    h, w, c = img.shape
    dy = (h-height)//2
    dx = (w-width)//2
    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]
    return img


def shift_hsv(img, hue_shift, sat_shift, val_shift):
    dtype = img.dtype
    maxval = np.max(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)
    h, s, v = cv2.split(img)
    h = cv2.add(h, hue_shift)
    h = np.where(h < 0, maxval - h, h)
    h = np.where(h > maxval, h - maxval, h)
    h = h.astype(dtype)
    s = clip(cv2.add(s, sat_shift), dtype, maxval)
    v = clip(cv2.add(v, val_shift), dtype, maxval)
    img = cv2.merge((h, s, v)).astype(dtype)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


def shift_channels(img, r_shift, g_shift, b_shift):
    img[...,0] = clip(img[...,0] + r_shift, np.uint8, 255)
    img[...,1] = clip(img[...,1] + g_shift, np.uint8, 255)
    img[...,2] = clip(img[...,2] + b_shift, np.uint8, 255)
    return img


def clahe(img, clipLimit=2.0, tileGridSize=(8,8)):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_LAB2RGB)
    return img_output


def blur(img, ksize):
    return cv2.blur(img, (ksize, ksize))


def invert(img):
    return 255 - img


def channel_shuffle(img):
    ch_arr = [0, 1, 2]
    np.random.shuffle(ch_arr)
    img = img[..., ch_arr]
    return img


def img_to_tensor(im, verbose=False):
    '''AVE edit'''
    im_out = np.moveaxis(im / (255. if im.dtype == np.uint8 else 1), -1, 0).astype(np.float32)
    if verbose:
        print ("augmentations.functiona.py.img_to_tensor(): im_out.shape:", im_out.shape)
        print ("im_out.unique:", np.unique(im_out))
    return im_out


def mask_to_tensor(mask, num_classes, verbose=False):
    '''AVE edit'''
    if num_classes > 1:
        mask = img_to_tensor(mask)
    else:
        mask = np.expand_dims(mask / (255. if mask.dtype == np.uint8 else 1), 0).astype(np.float32)
    if verbose:
        print ("augmentations.functiona.py.img_to_tensor(): mask.shape:", mask.shape)
        print ("mask.unique:", np.unique(mask))

    return mask


class Compose:
    """
    compose transforms from list to apply them sequentially
    """
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, **data):
        for t in self.transforms:
            data = t(**data)
        return data


class OneOf:
    """
    with probability prob choose one transform from list and apply it
    """
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            data = t(**data)
        return data


class BasicTransform:
    """
    base class for all transforms
    """
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, **kwargs):
        """
        override it if you need to apply different transforms to data
        for example you can define method apply_to_boxes and apply it to bounding boxes
        """
        if random.random() < self.prob:
            params = self.get_params()
            return {
                k: self.apply(a, **params) if k in self.targets else a
                for k, a in kwargs.items()
            }

        return kwargs

    def apply(self, img, **params):
        """
        override this method with transform you need to apply
        """
        raise NotImplementedError

    def get_params(self):
        """
        dict of transform parameters for apply
        """
        return {}

    @property
    def targets(self):
        raise NotImplementedError


class DualTransform(BasicTransform):
    """
    transfrom for segmentation task
    """
    @property
    def targets(self):
        return 'image', 'mask'


class ImageOnlyTransform(BasicTransform):
    """
    transforms applied to image only
    """
    @property
    def targets(self):
        return 'image'


class VerticalFlip(DualTransform):
    def apply(self, img, **params):
        return F.vflip(img)


class HorizontalFlip(DualTransform):
    def apply(self, img, **params):
        return F.hflip(img)


class RandomFlip(DualTransform):
    def apply(self, img, flipCode=0):
        return F.flip(img, flipCode)

    def get_params(self):
        return {'flipCode': random.randint(-1, 1)}


class Transpose(DualTransform):
    def apply(self, img, **params):
        return F.transpose(img)


class RandomRotate90(DualTransform):
    def apply(self, img, times=0):
        return F.rot90(img, times)

    def get_params(self):
        return {'times': random.randint(0, 4)}


class RandomRotate(DualTransform):
    def __init__(self, angle_limit=90, prob=.5):
        super().__init__(prob)
        self.angle_limit = angle_limit

    def apply(self, img, angle=0):
        return F.rotate(img, angle)

    def get_params(self):
        return {'angle': random.uniform(-self.angle_limit, self.angle_limit)}


class RandomShiftScaleRotate(DualTransform):
    def __init__(self,
                 shift_limit=0.0625,
                 scale_limit=0.1,
                 rotate_limit=45,
                 prob=0.5):
        super().__init__(prob)
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit

    def apply(self, img, angle=0, scale=0, dx=0, dy=0):
        return F.shift_scale_rotate(img, angle, scale, dx, dy)

    def get_params(self):
        return {
            'angle': random.uniform(-self.rotate_limit, self.rotate_limit),
            'scale': random.uniform(1-self.scale_limit, 1+self.scale_limit),
            'dx': round(random.uniform(-self.shift_limit, self.shift_limit)),
            'dy': round(random.uniform(-self.shift_limit, self.shift_limit)),
        }


class CenterCrop(DualTransform):
    def __init__(self, height, width, prob=0.5):
        super().__init__(prob)
        self.height = height
        self.width = width

    def apply(self, img, **params):
        return F.center_crop(img, self.height, self.width)


class Jitter_HSV(ImageOnlyTransform):
    def __init__(self,
                 hue_shift_limit=(-20, 20),
                 sat_shift_limit=(-35, 35),
                 val_shift_limit=(-35, 35),
                 prob=0.5):
        super().__init__(prob)
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit

    def apply(self, image, hue_shift=0, sat_shift=0, val_shift=0):
        hue_shift, sat_shift, val_shift = F.fix_shift_values(
            image,
            hue_shift,
            sat_shift,
            val_shift,
        )
        return F.shift_hsv(image, hue_shift, sat_shift, val_shift)

    def get_params(self):
        return {
            'hue_shift': np.random.uniform(self.hue_shift_limit[0],
                                           self.hue_shift_limit[1]),
            'sat_shift': np.random.uniform(self.sat_shift_limit[0],
                                           self.sat_shift_limit[1]),
            'val_shift': np.random.uniform(self.val_shift_limit[0],
                                           self.val_shift_limit[1]),
        }


class Jitter_channels(ImageOnlyTransform):
    def __init__(self,
                 r_shift_limit=(-20, 20),
                 g_shift_limit=(-20, 20),
                 b_shift_limit=(-20, 20),
                 prob=0.5):
        super().__init__(prob)
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit

    def apply(self, image, r_shift=0, g_shift=0, b_shift=0):
        r_shift, g_shift, b_shift = F.fix_shift_values(image, r_shift, g_shift, b_shift)
        return F.shift_channels(image, r_shift, g_shift, b_shift)

    def get_params(self):
        return {
            'r_shift': np.random.uniform(self.r_shift_limit[0],
                                         self.r_shift_limit[1]),
            'g_shift': np.random.uniform(self.g_shift_limit[0],
                                         self.g_shift_limit[1]),
            'b_shift': np.random.uniform(self.b_shift_limit[0],
                                         self.b_shift_limit[1]),
        }


class RandomBlur(ImageOnlyTransform):
    def __init__(self, blur_limit=7, prob=.5):
        super().__init__(prob)
        self.blur_limit = blur_limit

    def apply(self, image, ksize=3):
        return F.blur(image, ksize)

    def get_params(self):
        return {
            'ksize': np.random.choice(np.arange(3, self.blur_limit + 1, 2))
        }


class RandomCLAHE(ImageOnlyTransform):
    def __init__(self, clipLimit=4.0, tileGridSize=(8, 8), prob=0.5):
        super().__init__(prob)
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize

    def apply(self, img, clipLimit=2):
        return F.clahe(img, clipLimit, self.tileGridSize)

    def get_params(self):
        return {"clipLimit": np.random.uniform(1, self.clipLimit)}

class ChannelShuffle(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.channel_shuffle(img)

class InvertImg(ImageOnlyTransform):
    def apply(self, img, **params):
        return F.invert(img)

class ToTensor(BasicTransform):
    def __init__(self, num_classes=1):
        super().__init__(prob=1.)
        self.num_classes = num_classes

    def __call__(self, **kwargs):
        kwargs.update({
            'image': F.img_to_tensor(kwargs['image']),
        })

        if 'mask' in kwargs:
            kwargs.update({
                'mask': F.mask_to_tensor(kwargs['mask'], self.num_classes),
            })
        return kwargs


def get_flips_colors_augmentation(prob=.5):
    """
    you can compose transforms and apply them sequentially
    """
    return Compose([
        RandomFlip(0.5),
        Transpose(0.5),
        RandomShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.10,
                               rotate_limit=30,
                               prob=.75),
        Jitter_HSV()
    ])

def get_flips_shifts_augmentation(prob=.5):
    """
        you can compose transforms and apply them sequentially
        """
    return Compose([
        RandomFlip(0.5),
        Transpose(0.5),
        RandomShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.10,
                               rotate_limit=30,
                               prob=.75)
      ])
