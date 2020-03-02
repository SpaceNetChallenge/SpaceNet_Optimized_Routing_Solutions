import os
import argparse
import warnings

from tqdm import tqdm
import skimage
import cv2
import gdal
import numpy as np

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--unet-path',
        type=str,
        required=True,
        help='Path to 8-bit masks',
    )
    parser.add_argument(
        '--fpn-path',
        type=str,
        required=True,
        help='Path to 8-bit masks',
    )
    parser.add_argument(
        '--out-path',
        type=str,
        required=True,
        help='Path to output',
    )
    return parser.parse_args()


def CreateMultiBandGeoTiff(path: str, array: np.ndarray):
    '''
    Array has shape:
        Channels, Y, X?
    '''
    driver = gdal.GetDriverByName('GTiff')
    array = array.transpose(2, 0, 1)
    DataSet = driver.Create(path, array.shape[2], array.shape[1],
                            array.shape[0], gdal.GDT_Byte,
                            ['COMPRESS=LZW'])
    for i, image in enumerate(array, 1):
        DataSet.GetRasterBand(i).WriteArray(image)
    del DataSet
    img = skimage.io.imread(path)
    # print(img.shape)
    assert img.shape[1] == img.shape[2] == 1300

    return array

def main():
    args = parse_args()
    unet_paths = sorted(os.listdir(args.unet_path))
    print(f'Found {len(unet_paths)} masks')
    fpn_paths = sorted(os.listdir(args.fpn_path))
    # assert(len(unet_paths) == len(fpn_paths))
    print(f"Reading masks from {args.unet_path} and {args.fpn_path}")
    for unet_path in tqdm(unet_paths, desc="Converting masks...", total=len(unet_paths)):
        unet_mask = cv2.imread(os.path.join(args.unet_path, unet_path), cv2.IMREAD_GRAYSCALE)
        fpn_path = unet_path.replace('.png', '.tif')
        if not os.path.exists(os.path.join(args.fpn_path, fpn_path)):
            fpn_path = fpn_path.replace('SN', 'fold0_SN')
        fpn_mask = skimage.io.imread(os.path.join(args.fpn_path, fpn_path))
        if fpn_mask.shape[0] == fpn_mask.shape[1] == 1300 and fpn_mask.ndim == 3:
            fpn_mask = fpn_mask.transpose((2, 0, 1))

        # print(unet_mask.shape, np.max(fpn_mask, axis=2).shape)

        # print(fpn_mask.shape)
        # mask = skimage.io.imread(os.path.join(args.in_path, mask_path))
        assert unet_mask.shape[1] == unet_mask.shape[0] == 1300
        assert unet_mask.ndim == 2
        assert fpn_mask.ndim == 3
        assert fpn_mask.shape[1] == fpn_mask.shape[2] == 1300

        fpn_mask[-1, ...] = unet_mask
        # mask_name = mask_path.split("/")[-1]
        save_path = fpn_path
        CreateMultiBandGeoTiff(os.path.join(args.out_path, save_path), fpn_mask)
        # skimage.io.imsave(os.path.join(args.out_path, save_path), fpn_mask)
    print(f"Saved to {args.out_path}")


if __name__ == '__main__':
    main()
