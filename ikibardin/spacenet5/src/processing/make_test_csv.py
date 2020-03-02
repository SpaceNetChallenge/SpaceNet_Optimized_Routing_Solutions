import os
import argparse

import pandas as pd
import power_fist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--out-csv',
        type=str,
        required=True,
        help='Where to save .csv dataframe with folds split',
    )
    parser.add_argument(
        '--paths',
        type=str,
        default='configs/paths_default.yml',
        help='Path to config file with paths for dataset and logs',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths_config = power_fist.config_utils.get_paths(args.paths)
    test_images_folder = paths_config['dataset']['test_dir']

    image_ids = [os.path.splitext(filename)[0] for filename in os.listdir(test_images_folder)]
    df = pd.DataFrame(image_ids, columns=['id'])
    print(df.head())
    df.to_csv(args.out_csv, index=False)
    print(f'Saved folds dataframe of shape {df.shape} to `{args.out_csv}`')


if __name__ == '__main__':
    main()
