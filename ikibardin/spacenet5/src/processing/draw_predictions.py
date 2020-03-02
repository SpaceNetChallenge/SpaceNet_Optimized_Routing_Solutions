import os
import argparse
import multiprocessing
from typing import Tuple, Dict, Union, Iterable

import numpy as np
import cv2
import skimage.io
from tqdm import tqdm


class Visualizer:
    def __init__(
            self, images_dir: str, gt_masks_dir: Union[str, None], pred_masks_dir: str, output_dir: str,
    ):
        self._images_dir = images_dir
        self._gt_masks_dir = gt_masks_dir
        self._pred_masks_dir = pred_masks_dir

        os.makedirs(output_dir, exist_ok=True)
        self._output_dir = output_dir

    def dump_visualization(self, image_id: str):
        visualization = self.get_visualization(image_id)
        cv2.imwrite(
            os.path.join(self._output_dir, f'{image_id}.jpg'),
            cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR),
        )

    def get_visualization(self, image_id: str) -> np.ndarray:
        image = self._load_image(image_id)

        if self._gt_masks_dir is not None:
            gt_vis = self._get_visualization_piece(image, self._gt_masks_dir, image_id)
        else:
            gt_vis = image

        pred_vis = self._get_visualization_piece(image, self._pred_masks_dir, image_id)

        return np.vstack([gt_vis, pred_vis])

    def _get_visualization_piece(self, image: np.ndarray, masks_dir: str, image_id: str) -> np.ndarray:
        mask = self._load_mask(masks_dir, image_id)
        return self._apply_mask(image, mask, color=(0, 255, 0))

    def _load_image(self, image_id: str) -> np.ndarray:
        # print(image_id)
        image_id = image_id.replace('.png', '')
        path = os.path.join(self._images_dir, f'{image_id}.tif')

        if not os.path.exists(path):
            raise ValueError(f'Image not found at `{path}`')
        image = skimage.io.imread(path)
        assert image is not None, path
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_mask(self, masks_dir: str, image_id: str) -> np.ndarray:
        path = os.path.join(masks_dir, f'{image_id}.png')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        assert mask is not None, path
        mask = mask / 255.0
        return mask

    @staticmethod
    def _apply_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha=0.5) -> np.ndarray:
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f'Image and mask shape mismatch: {image.shape} vs {mask.shape}')
        output = image.copy()
        overlay = output.copy()
        overlay[mask > 0.5] = color

        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to directory with raw images',
    )
    parser.add_argument(
        '--gt-masks',
        type=str,
        default=None,
        help='Path to directory with gt masks',
    )
    parser.add_argument(
        '--pred-masks',
        type=str,
        required=True,
        help='Path to directory with predicted masks',
    )
    parser.add_argument(
        '--out-dir',
        type=str,
        required=True,
        help='Where to save visualizations',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=multiprocessing.cpu_count(),
        help='Number of threads to use',
    )
    parser.set_defaults(add_metrics=False, )
    return parser.parse_args()


def main():
    args = parse_args()

    image_ids = [os.path.splitext(p)[0] for p in os.listdir(args.pred_masks)]
    # image_ids = sorted(list(filter(
    #     lambda id_: 'dehuman' not in id_,
    #     map(
    #         get_image_id,
    #         glob.glob(os.path.join(args.images, '*')),
    #     ),
    # )))
    print(f'Found {len(image_ids)} image ids')

    visualizer = Visualizer(
        images_dir=args.images,
        gt_masks_dir=args.gt_masks,
        pred_masks_dir=args.pred_masks,
        output_dir=args.out_dir,
    )
    with multiprocessing.Pool(args.num_workers) as pool:
        list(
            tqdm(
                pool.imap_unordered(visualizer.dump_visualization, image_ids),
                total=len(image_ids),
                desc=f'Dumping visualization to `{args.out_dir}`',
            )
        )


if __name__ == '__main__':
    main()
