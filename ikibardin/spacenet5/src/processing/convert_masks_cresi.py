import os
import argparse
import warnings

from tqdm import tqdm
import skimage
import cv2

warnings.filterwarnings('ignore')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in-path',
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


def main():
    args = parse_args()
    masks_paths = os.listdir(args.in_path)
    print(f"Reading masks from {args.in_path}")
    for mask_path in tqdm(masks_paths, desc="Converting masks..."):
        mask = cv2.imread(os.path.join(args.in_path, mask_path), cv2.IMREAD_GRAYSCALE)
        # mask = skimage.io.imread(os.path.join(args.in_path, mask_path))
        assert mask.shape[1] == mask.shape[0] == 1300
        assert mask.ndim == 2
        # mask_name = mask_path.split("/")[-1]
        save_path = 'fold0_' + mask_path.replace('PS-RGB', 'PS-MS').replace('png', 'tif')
        skimage.io.imsave(os.path.join(args.out_path, save_path), mask)
    print(f"Saved to {args.out_path}")


if __name__ == '__main__':
    main()
