import glob
import os
from tqdm import tqdm
import argparse

import geopandas as gpd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--geojsons',
        type=str,
        required=True,
        help='Path to a directory with geojsons folders. Structure of the folder: */*/*.geojson',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geojsons = glob.glob(os.path.join(args.geojsons, '*', '*', '*.geojson'))
    dataset_speeds = []
    for geojson in tqdm(geojsons):
        df = gpd.read_file(geojson)
        img_speeds = df['inferred_speed_mps'].values.tolist()
        dataset_speeds += img_speeds
    print(f'Mean speed = {np.mean(dataset_speeds)} mps')


if __name__ == "__main__":
    main()
