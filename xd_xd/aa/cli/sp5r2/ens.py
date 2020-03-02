import time
from pathlib import Path
from logging import getLogger
import json
import argparse
import os
import sys

import skimage.io
import pandas as pd
import numpy as np
import click
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
from tqdm import tqdm

if torch.cuda.is_available():
    torch.randn(10).cuda()

tqdm.monitor_interval = 0

# todo: make this better
sys.path.append('./')  # aa

from aa.cresi.net.pytorch_utils.concrete_eval import FullImageEvaluator
from aa.road_networks.skeletonize import run_skeletonize
from aa.road_networks.cleaning_graph import cleaning_graph
from aa.road_networks.infer_speed import infer_speed
from aa.road_networks.create_submission import make_sub, make_sub_debug
from aa.pytorch.transforms import get_flips_colors_augmentation
from aa.pytorch.data_provider import ReadingImageProvider, RawImageType
import aa.cli.sp5r2.util as u


logger = getLogger('aa')


@click.group()
def cli():
    pass


@cli.command()
@click.option('-c', '--config_path', type=str)
def ensemble(config_path):
    # 03a
    conf = u.load_config(config_path)
    u.set_filehandler(conf)

    logger.info('ARGV: {}'.format(str(sys.argv)))
    ensemble_merge_folds(conf)

    # Output: ske, sknw_gpickle, wkt
    # Required time with single process: 30min -> (multiproc: 5min)
    run_skeletonize(conf)

    # Output: graphs
    # Required time with single process: 7min -> (multiproc: 2min)
    cleaning_graph(conf)

    # Output: graphs_speed
    # Required time with single process: 2min
    infer_speed(conf)

    # Output: solution
    # Required time with single process: 0min
    make_sub(conf)
    make_sub_debug(conf)


@cli.command()
@click.option('-c', '--config_path', type=str)
def ensemble_th06(config_path):
    # 03a
    conf = u.load_config(config_path)
    u.set_filehandler(conf)

    logger.info('ARGV: {}'.format(str(sys.argv)))

    conf.modelname = conf.modelname + '_th06'
    conf.skeleton_thresh = 0.6

    run_skeletonize(conf)
    cleaning_graph(conf)
    infer_speed(conf)
    make_sub(conf)
    make_sub_debug(conf)


@cli.command()
@click.option('-i', '--input-path', type=str)
@click.option('-o', '--output-path', type=str)
def remove_small_edges(input_path, output_path):
    length_thresh = 100
    pix_thresh = 1000

    remove_ids = []
    df = pd.read_csv(input_path)
    df_agg = df.groupby('ImageId').agg('sum').reset_index()
    for _, r in df_agg[(df_agg.length_m > 0) & (df_agg.length_m < length_thresh)].sort_values(by='length_m').head(30).iterrows():
        imdir = Path('/wdata/working/sp5r2/models/preds/ens_r50a_serx50/merged_test/')
        aoi_name = '_'.join(r.ImageId.split('_')[:-1])
        chip_name = r.ImageId.split('_')[-1]
        path = imdir / ('SN5_roads_test_private_' + aoi_name + '_PS-MS_' + chip_name + '.tif')
        if not path.exists():
            path = imdir / ('SN5_roads_test_public_' + aoi_name + '_PS-MS_' + chip_name + '.tif')

        im = skimage.io.imread(str(path))
        im2 = np.sum(im, axis=0).astype(np.int8)
        if (im2 > (255 * 0.4)).sum() < pix_thresh:
            print(path.name, r.length_m, (im2 > (255 * 0.4)).sum(), (im2 > (255 * 0.5)).sum(), (im2 > (255 * 0.6)).sum())
            remove_ids.append(r.ImageId)

    df_1 = df[pd.np.logical_not(df.ImageId.isin(remove_ids))]
    df_2 = pd.DataFrame([{'ImageId': image_id, 'WKT_Pix': "LINESTRING EMPTY", 'length_m': 0.0, 'travel_time_s': 0.0, 'speed_mph': 0.0} for image_id in remove_ids])
    df_ = pd.concat([df_1, df_2], sort=False)
    df_[['ImageId', 'WKT_Pix', 'length_m', 'travel_time_s']].to_csv(output_path, index=False)


def ensemble_merge_folds(conf):
    first_modelname = conf.ensemble_folds[0]['name']
    fold0_pred_dir = "/wdata" + f'/working/sp5r2/models/preds/{first_modelname}/fold0_test/'
    files = sorted(Path(fold0_pred_dir).glob('./*.tif'))

    # TODO: multiprocess
    for fn in tqdm(files, total=len(files)):
        preds = []
        for ens_part in conf.ensemble_folds:
            for fold_idx in range(ens_part.nfolds):
                fn_elem = Path(
                    "/wdata" +
                    f'/working/sp5r2/models/preds/{ens_part.name}/fold0_test/'
                )

                name = f'fold{fold_idx}_' + fn.name.lstrip('fold0_')
                model_pred_base = str(fn_elem.parent / f'fold{fold_idx}_test' / name)
                pred = skimage.io.imread(model_pred_base)
                preds.append(pred)
        preds = np.mean(preds, axis=0).astype(np.uint8)

        merged_path = str(
            Path("/wdata" + f'/working/sp5r2/models/preds/{conf.modelname}') / 'merged_test' / fn.name.lstrip('fold0_'))
        print(merged_path)
        Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(merged_path, preds, compress=1)


if __name__ == '__main__':
    u.set_logger()
    cli()
