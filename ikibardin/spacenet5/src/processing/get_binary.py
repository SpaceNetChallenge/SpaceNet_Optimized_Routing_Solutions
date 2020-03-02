import os
import glob
import argparse
import skimage.io
import numpy as np
import cv2
from tqdm import tqdm

TIFS = '/home/ikibardin/dataset/spacenet5/a'
OUT = '/home/ikibardin/dataset/spacenet5/b'


def main():
    os.makedirs(OUT, exist_ok=True)
    tif_paths = glob.glob(os.path.join(TIFS, '*'))
    for p in tqdm(tif_paths):
        mask = skimage.io.imread(p)
        print(mask.shape)
        mask = mask[:, :, 1]
        print(mask.min(), mask.max(), np.unique(mask))
        cv2.imwrite(
            os.path.join(OUT, os.path.splitext(os.path.basename(p))[0] + '.png'),
            mask,
        )


if __name__ == "__main__":
    main()
