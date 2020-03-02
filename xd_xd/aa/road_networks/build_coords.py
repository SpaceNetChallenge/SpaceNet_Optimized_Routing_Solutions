from pathlib import Path

import tqdm
import numpy as np
import pandas as pd
from osgeo import gdal


AOI_NAMES = [
    'AOI_7_Moscow',
    'AOI_8_Mumbai',
    'AOI_9_San_Juan',
]


def get_coords(aoi_name, data_folder):
    rows = []

    image_dir = (Path(data_folder) / "PS-MS")
    n_images = len(list(image_dir.glob('./*.tif')))
    if n_images == 0:
        raise RuntimeError(f"Can't find GeoTIFF images: data_folder={data_folder}")

    for path in image_dir.glob('./*.tif'):
        imname = path.stem.split('_')[-1]  # e.g. "chip100"

        src = gdal.Open(str(path))
        ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
        lrx = ulx + (src.RasterXSize * xres)
        lry = uly + (src.RasterYSize * yres)
        rows.append({
            'aoi_name': aoi_name,
            'imname': imname,
            'mode': 'test',
            'lrx': lrx,
            'ulx': ulx,
            'lry': lry,
            'uly': uly,
        })

    return pd.DataFrame(rows)


def main(aoi_data_path_mapping):
    df_list = []
    for aoi_name in sorted(aoi_data_path_mapping.keys()):
        df = get_coords(aoi_name, aoi_data_path_mapping[aoi_name])
        if len(df) == 0:
            raise RuntimeError("Failed to parse AOI coordinates")

        df['ulx'] = df['ulx'] * 1000 * 1000
        df['lrx'] = df['lrx'] * 1000 * 1000
        df['uly'] = df['uly'] * 1000 * 1000
        df['lry'] = df['lry'] * 1000 * 1000

        xdiff = df.sort_values(by='ulx').ulx.drop_duplicates().diff().dropna().value_counts().index[0]
        xmin = df.sort_values(by='ulx').ulx.drop_duplicates().min()
        df['ix'] = ((df.ulx - xmin) / xdiff).apply(lambda x: round(x))
        assert df.groupby('ix').ulx.agg(['min', 'max', 'nunique'])['nunique'].max() == 1

        ydiff = df.sort_values(by='uly').uly.drop_duplicates().diff().dropna().value_counts().index[0]
        ymin = df.sort_values(by='uly').uly.drop_duplicates().min()
        df['iy'] = ((df.uly - ymin) / ydiff).apply(lambda x: round(x))
        df['iy'] = df['iy'].max() - df['iy']
        assert df.groupby('iy').uly.agg(['min', 'max', 'nunique'])['nunique'].max() == 1

        df_list.append(df)

    df = pd.concat(df_list, sort=False)
    # df.to_csv(out_chiplocations, index=False)
    return df


if __name__ == '__main__':
    # out_chiplocations = 'data/working/sn5/stat/AOI_all_bigmap_chip_locations.csv'
    df = main(aoi_data_path_mapping)
    assert len(df) > 0
