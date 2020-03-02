import json
import sys
import os
from logging import getLogger
from pathlib import Path

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import click
import torch
import pandas as pd
import numpy as np

# todo: make this better
sys.path.append('./')  # aa

from aa.pytorch.data_provider import ReadingImageProvider, RawImageType
from aa.pytorch.transforms import get_flips_colors_augmentation
import aa.pytorch.trainer
import aa.cli.sp5r2.util as u

logger = getLogger('aa')


torch.randn(10).cuda()


class RawImageTypePad(RawImageType):
    def finalyze(self, data):
        padding_size = 22
        return self.reflect_border(data, padding_size)


def train(conf, args_fold):
    paths = {
        'masks': conf.train_data_refined_dir_masks,
        'images': conf.train_data_refined_dir_ims,
    }
    fn_mapping = {
        'masks': lambda name: os.path.splitext(name)[0] + '.tif',
    }

    ds = ReadingImageProvider(RawImageType,
                              paths,
                              fn_mapping,
                              image_suffix='',
                              num_channels=conf.num_channels)
    val_ds = None
    logger.info(f'Total Dataset size: {len(ds)}')

    fn_val_splits = conf.folds_save_path
    folds = u.get_csv_folds(fn_val_splits, ds.im_names)
    for fold, (train_idx, val_idx) in enumerate(folds):
        if int(args_fold) != fold:
            continue

        logger.info(f'Fold idx: {args_fold}')
        logger.info(f'Train size: {len(train_idx)}')
        logger.info(f'Val size: {len(val_idx)}')
        transforms = get_flips_colors_augmentation()
        aa.pytorch.trainer.train(ds,
                                 fold,
                                 train_idx,
                                 val_idx,
                                 conf,
                                 transforms=transforms,
                                 val_ds=val_ds)


@click.command()
@click.option('-c', '--config_path', type=str)
@click.option('-f', '--fold', type=int, default=0)
def main(config_path, fold):
    conf = u.load_config(config_path)
    u.set_filehandler(conf)

    logger.info('ARGV: {}'.format(str(sys.argv)))
    train(conf, fold)


if __name__ == '__main__':
    u.set_logger()
    main()
