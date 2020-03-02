import time
from pathlib import Path
from logging import getLogger
import json
import argparse
import os
import sys

import skimage.io
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
@click.option('-f', '--fold', type=int, default=0)
def evaltest(config_path, fold):
    # 02
    conf = u.load_config(config_path)
    u.set_filehandler(conf)

    logger.info('ARGV: {}'.format(str(sys.argv)))
    eval_test(conf, fold, nfolds=1)


@cli.command()
@click.option('-c', '--config_path', type=str)
def mergefolds(config_path):
    # 03a
    conf = u.load_config(config_path)
    u.set_filehandler(conf)

    logger.info('ARGV: {}'.format(str(sys.argv)))
    merge_folds(conf, nfolds=conf.num_folds)


@cli.command()
@click.option('-c', '--config_path', type=str)
def makegraph(config_path):
    # 04, 05, 06, 07: skelton, simplify, infer_speed, sub
    conf = u.load_config(config_path)
    u.set_filehandler(conf)

    logger.info('ARGV: {}'.format(str(sys.argv)))

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


class RawImageTypePad(RawImageType):
    def finalyze(self, data):
        padding_size = 22
        return self.reflect_border(data, padding_size)


def eval_test(conf, args_fold, nfolds=1):
    # Full resolution input
    conf.target_rows = conf.eval_rows
    conf.target_cols = conf.eval_cols

    paths = {
        'masks': '',
        'images': conf.test_data_refined_dir_ims,
    }
    fn_mapping = {
        'masks': lambda name: os.path.splitext(name)[0] + '.tif',
    }
    ds = ReadingImageProvider(RawImageTypePad,
                              paths,
                              fn_mapping,
                              image_suffix='',
                              num_channels=conf.num_channels)

    logger.info(f'Total Dataset size: {len(ds)}')
    folds = [([], list(range(len(ds)))) for i in range(nfolds)]

    save_dir = f'/wdata/working/sp5r2/models/preds/{conf.modelname}/fold{args_fold}_test'
    weight_dir = f'/wdata/working/sp5r2/models/weights/{conf.modelname}/fold{args_fold}'
    keval = FullImageEvaluator(conf,
                               ds,
                               save_dir=save_dir,
                               test=True,
                               flips=3,
                               num_workers=2,
                               border=conf.padding,
                               save_im_gdal_format=False)
    for fold, (train_idx, test_idx) in enumerate(folds):
        logger.info(f'Fold idx: {args_fold}')
        logger.info(f'Train size: {len(train_idx)}')
        logger.info(f'Test size: {len(test_idx)}')
        keval.predict(args_fold, test_idx, weight_dir, verbose=False)
        break


def merge_folds(conf, nfolds=4):
    fold0_pred_dir = f'/wdata/working/sp5r2/models/preds/{conf.modelname}/fold0_test/'
    files = sorted(Path(fold0_pred_dir).glob('./*.tif'))

    # TODO: multiprocess
    for fn in tqdm(files, total=len(files)):
        preds = []
        for fold_idx in range(nfolds):
            name = f'fold{fold_idx}_' + fn.name.lstrip('fold0_')
            model_pred_base = str(fn.parent.parent / f'fold{fold_idx}_test' / name)
            pred = skimage.io.imread(model_pred_base)
            preds.append(pred)
        preds = np.mean(preds, axis=0).astype(np.uint8)

        merged_path = str(fn.parent.parent / 'merged_test' / fn.name.lstrip('fold0_'))
        Path(merged_path).parent.mkdir(parents=True, exist_ok=True)
        skimage.io.imsave(merged_path, preds, compress=1)


if __name__ == '__main__':
    u.set_logger()
    cli()
