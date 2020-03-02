import os
import argparse
import multiprocessing
from typing import Dict, Union

import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm

from src import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--masks',
        type=str,
        required=True,
        help='Path to a directory with training masks',
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        required=True,
        help='Where to save .csv dataframe with folds split',
    )
    return parser.parse_args()


class MetadataGetter:
    def __init__(self, masks_dir: str):
        self._masks_dir = masks_dir

    def get_metadata(self, image_id: str) -> Dict[str, Union[str, float]]:
        mask = self._load_mask(image_id) > 127
        return {
            'id': image_id,
            'city_id': self._get_city_id(image_id),
            'roads_ratio': mask.sum() / mask.size,
        }

    def _load_mask(self, image_id: str) -> np.ndarray:
        path = os.path.join(self._masks_dir, f'{image_id}.tif')
        if not os.path.exists(path):
            raise ValueError(path)
        mask = skimage.io.imread(path)
        assert mask is not None, path
        # print('SHAPE', mask.shape)
        mask = mask[:, :, -1]
        return mask

    @staticmethod
    def _get_city_id(image_id: str) -> str:
        city_id = '_'.join(image_id.split('_')[3:6])
        assert city_id in config.TRAINING_CITIES, (city_id, config.TRAINING_CITIES)
        return city_id


def get_folds_split(df: pd.DataFrame, num_folds: int = 5) -> pd.DataFrame:
    df = df.sample(n=len(df), replace=False)
    df = df.sort_values(by=['city_id', 'roads_ratio']).reset_index(drop=True)
    df['fold_id'] = np.arange(len(df)) % num_folds
    return df


def main():
    args = parse_args()
    image_ids = [os.path.splitext(filename)[0] for filename in os.listdir(args.masks)]

    with multiprocessing.Pool(16) as pool:
        metadata = list(tqdm(
            pool.imap_unordered(MetadataGetter(args.masks).get_metadata, image_ids),
            total=len(image_ids),
            desc='Extracting metadata...',
        ))

    df = pd.DataFrame(metadata)
    df = get_folds_split(df)
    df.to_csv(args.out_csv, index=False)
    print(f'Saved folds dataframe of shape {df.shape} to `{args.out_csv}`')


if __name__ == '__main__':
    main()
