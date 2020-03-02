import os
import random

import pandas as pd
import numpy as np
from easydict import EasyDict as edict


def make_split(fold_seed=2434, fold_suffix='b'):
    config = edict(
        train_data_refined_dir_ims='/wdata/input/train/images_8bit_base/PS-RGB/',
        folds_save_path=f'/wdata/input/train/folds_{fold_suffix}.csv',
        default_val_perc=0.20,
        num_folds=5,
    )
    random.seed(fold_seed)

    ims_files = os.listdir(config.train_data_refined_dir_ims)
    if os.path.exists(config.folds_save_path):
        print("folds csv already exists:", config.folds_save_path)
        return
    else:
        print ("folds_save_path:", config.folds_save_path)
        random.shuffle(ims_files)
        s = {k.split('_')[0] for k in ims_files}
        d = {k: [v for v in ims_files] for k in s}

        folds = {}

        if config.num_folds == 1:
            nfolds = int(np.rint(1. / config.default_val_perc))
        else:
            nfolds = config.num_folds

        idx = 0
        for v in d.values():
            for val in v:
                folds[val] = idx % nfolds
                idx+=1

        df = pd.Series(folds, name='fold')
        df.to_csv(config.folds_save_path, header=['fold'], index=True)


if __name__ == "__main__":
    make_split(fold_seed=2434, fold_suffix='a')
