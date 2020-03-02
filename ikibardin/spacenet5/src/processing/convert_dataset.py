import os
import glob
import argparse
import multiprocessing
import subprocess
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2
import tifffile
import pygeoif
import gdal
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to directory with training images',
    )
    parser.add_argument(
        '--roads',
        type=str,
        required=False,
        default=None,
        help='Path to .csv with roads',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Where to save converted dataset',
    )
    return parser.parse_args()


def load_roads(csv_path: str) -> Dict[str, List[str]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    gt = {}
    matrix = pd.read_csv(csv_path).as_matrix()
    for line in matrix:
        id = line[0]
        linestring = line[1]
        gt_lines = gt.get(id, [])
        gt_lines.append(linestring)
        gt[id] = gt_lines
    return gt


def convert_to_8Bit(inputRaster, outputRaster,
                    outputPixType='Byte',
                    outputFormat='GTiff',
                    rescale_type='rescale',
                    percentiles=[2, 98]):
    '''
    Convert 16bit image to 8bit
    rescale_type = [clip, rescale]
        if clip, scaling is done strictly between 0 65535
        if rescale, each band is rescaled to a min and max
        set by percentiles
    '''

    srcRaster = gdal.Open(inputRaster)
    cmd = ['gdal_translate', '-ot', outputPixType, '-of',
           outputFormat]

    # iterate through bands
    for bandId in range(srcRaster.RasterCount):
        bandId = bandId + 1
        band = srcRaster.GetRasterBand(bandId)
        if rescale_type == 'rescale':
            bmin = band.GetMinimum()
            bmax = band.GetMaximum()
            # if not exist minimum and maximum values
            if bmin is None or bmax is None:
                (bmin, bmax) = band.ComputeRasterMinMax(1)
            # else, rescale
            band_arr_tmp = band.ReadAsArray()
            bmin = np.percentile(band_arr_tmp.flatten(),
                                 percentiles[0])
            bmax = np.percentile(band_arr_tmp.flatten(),
                                 percentiles[1])

        else:
            bmin, bmax = 0, 65535

        cmd.append('-scale_{}'.format(bandId))
        cmd.append('{}'.format(bmin))
        cmd.append('{}'.format(bmax))
        cmd.append('{}'.format(0))
        cmd.append('{}'.format(255))

    cmd.append(inputRaster)
    cmd.append(outputRaster)
    # print("Conversin command:", cmd)
    subprocess.call(cmd)

    return


class DatasetConverter:
    def __init__(self, roads_csv: Optional[str], output_dir: str, thickness: int = 20):
        if roads_csv is None:
            self._mode = 'test'
        else:
            self._mode = 'train'

        if self._mode == 'train':
            self._roads_data = load_roads(roads_csv)

            self._masks_output_dir = os.path.join(output_dir, 'masks')
            os.makedirs(self._masks_output_dir, exist_ok=True)

            self._visualization_dir = os.path.join(output_dir, 'visualized')
            os.makedirs(self._visualization_dir, exist_ok=True)

            self._thickness = thickness

        self._images_output_dir = os.path.join(output_dir, 'images')
        os.makedirs(self._images_output_dir, exist_ok=True)

    def convert(self, image_path: str):
        image = self._load_image(image_path)

        cv2.imwrite(
            self._get_image_output_path(image_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

        if self._mode == 'train':
            mask = self._create_mask(self._get_image_id(image_path), image)
            cv2.imwrite(self._get_mask_output_path(image_path), mask)

            visualization = self._apply_mask(image, mask, color=(0, 255, 0))
            cv2.imwrite(self._get_visualization_output_path(image_path), visualization)

    def _load_image(self, path: str, invert_color: bool = False) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        image = tifffile.imread(path)
        assert image is not None, path
        assert isinstance(image, np.ndarray), type(image)

        if image.dtype == np.uint16:
            convert_to_8Bit(path, self._get_image_output_path(path))
            return self._load_image(self._get_image_output_path(path), invert_color=True)

        assert image.dtype == np.uint8, image.dtype
        if invert_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _create_mask(self, image_id: str, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        mask = np.zeros(shape=(h, w), dtype=np.uint8)

        for line in self._roads_data[image_id.replace('_PS-RGB', '').replace('SN3_roads_train_', '')]:
            if "LINESTRING EMPTY" == line:
                continue
            points = pygeoif.from_wkt(line).coords
            for i in range(1, len(points)):
                pt1 = (int(points[i - 1][0]), int(points[i - 1][1]))
                pt2 = (int(points[i][0]), int(points[i][1]))
                cv2.line(mask, pt1, pt2, 255, thickness=self._thickness)
        return mask

    def _get_image_output_path(self, image_path: str) -> str:
        return os.path.join(self._images_output_dir, self._get_image_id(image_path) + '.png')

    def _get_mask_output_path(self, image_path: str) -> str:
        return os.path.join(self._masks_output_dir, self._get_image_id(image_path) + '.png')

    def _get_visualization_output_path(self, image_path: str) -> str:
        return os.path.join(self._visualization_dir, self._get_image_id(image_path) + '.png')

    @staticmethod
    def _get_image_id(image_path: str) -> str:
        return os.path.splitext(os.path.basename(image_path))[0]

    @staticmethod
    def _apply_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha=0.5) -> np.ndarray:
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f'Image and mask shape mismatch: {image.shape} vs {mask.shape}')
        output = image.copy()
        overlay = output.copy()
        overlay[mask > 0.5] = color

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output


def main():
    args = parse_args()
    image_paths = glob.glob(os.path.join(args.images, '*'))

    converter = DatasetConverter(roads_csv=args.roads, output_dir=args.output)

    with multiprocessing.Pool(16) as pool:
        list(tqdm(
            pool.imap_unordered(converter.convert, image_paths),
            total=len(image_paths),
            desc='Converting dataset...',
        ))


if __name__ == '__main__':
    main()
