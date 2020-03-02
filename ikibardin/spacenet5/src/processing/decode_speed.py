import os
import glob
import argparse
import multiprocessing

import numpy as np
import cv2
from tqdm import tqdm

MEDIAN_SPEED_IN_BINS = np.array([5, 15, 25, 35, 45, 55, 65], dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--masks',
        type=str,
        required=True,
        help='Directory with predicted speed masks',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Where to save decoded speed masks',
    )
    return parser.parse_args()


class SpeedDecoder:
    def __init__(self, masks_dir: str, output_dir: str):
        self._masks_dir = masks_dir
        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

    def dump_visualization(self, image_id: str):
        speed_mask = self.get_speed_mask(image_id)
        speed_mask = (speed_mask / 65.0 * 255.0).astype(np.uint8)
        speed_visualization = cv2.applyColorMap(np.dstack([speed_mask, speed_mask, speed_mask]), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self._output_dir, f'{image_id}.png'), speed_visualization)

    def get_speed_mask(self, image_id: str) -> np.ndarray:
        binned_masks = []
        for mask_path in sorted(glob.glob(os.path.join(self._masks_dir, image_id, '*'))):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            assert mask is not None, mask_path
            binned_masks.append(mask / 255.0)
        binned_masks = np.stack(binned_masks)
        assert np.allclose(binned_masks.sum(axis=0), 1.0, rtol=0.1), \
            (binned_masks.sum(axis=0).min(), binned_masks.sum(axis=0).max())

        speed_mask = np.average(binned_masks, axis=0, weights=MEDIAN_SPEED_IN_BINS) * MEDIAN_SPEED_IN_BINS.sum()

        print(' >>> ', speed_mask.shape, speed_mask.min(), speed_mask.mean(), speed_mask.max())
        assert len(speed_mask.shape) == 2, speed_mask.shape
        return speed_mask


def main():
    args = parse_args()
    image_ids = os.listdir(args.masks)
    if len(image_ids) == 0:
        raise FileNotFoundError(f'No masks found in `{args.masks}`')
    print(f'Found {len(image_ids)} samples')
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        list(tqdm(
            pool.imap_unordered(SpeedDecoder(args.masks, args.output).dump_visualization, image_ids),
            total=len(image_ids),
            desc=f'Dumping speed masks visualization to `{args.output}`',
        ))


if __name__ == "__main__":
    main()
