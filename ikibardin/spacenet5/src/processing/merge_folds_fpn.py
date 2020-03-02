import os
import argparse
import warnings
import glob

from tqdm import tqdm
import skimage
import cv2
import gdal
import numpy as np

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folds-path',
        type=str,
        required=True,
        help='Path to masks of predicted folds',
    )
    parser.add_argument(
        '--out-path',
        type=str,
        required=True,
        help='Path to output',
    )
    parser.add_argument(
        '--folds-num',
        type=int,
        required=True,
        help='Number of folds'
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

    fold0_masks_paths = glob.glob(os.path.join(args.folds_path, 'fold0_*'))
    print(f'Found {len(fold0_masks_paths)} masks')
    print(f"Reading masks from {args.folds_path}")
    for fold0_path in tqdm(fold0_masks_paths, desc="Converting masks...", total=len(fold0_masks_paths)):
        # print('fold0', fold0_path)
        fold0_name = fold0_path.split('/')[-1]
        fold_masks = []
        for i in range(args.folds_num):
            fold_path = fold0_name.replace('fold0', f'fold{i}')
            fold_masks.append(skimage.io.imread(os.path.join(args.folds_path, fold_path)))

        fold_masks = np.array(fold_masks, dtype=float)
        fold_mask = np.mean(fold_masks, axis=0).astype(np.uint8)

        assert fold_mask.ndim == 3
        assert fold_mask.shape[1] == fold_mask.shape[2] == 1300

        save_path = fold0_name.replace('fold0_', '')

        CreateMultiBandGeoTiff(os.path.join(args.out_path, save_path), fold_mask)

    print(f"Saved to {args.out_path}")


if __name__ == '__main__':
    main()
