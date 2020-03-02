import os
import sys
import datetime

from pathlib import Path
from logging import getLogger, Formatter, StreamHandler, INFO, WARNING
from logging import FileHandler
import importlib

import pandas as pd
import numpy as np
from easydict import EasyDict as edict


def is_devmode():
    return True if os.uname().nodename == 'resona' else False


def prefix_path():
    if not is_devmode():
        return '/data'
    else:
        return 'data'


LOGDIR = '/wdata/working/sp5r2/models/logs/{modelname:s}'


def load_config(config_path):
    mod = importlib.import_module(config_path.rstrip('.py').replace('/', '.'))
    return edict(mod.CONFIG)


def set_filehandler(conf, prefix='train'):
    logformat = '%(asctime)s %(levelname)s %(message)s'

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')[2:]
    logfile_path = str(
        Path(LOGDIR.format(modelname=conf.modelname)) /
        f'{prefix:s}_{timestamp:s}.log')
    Path(logfile_path).parent.mkdir(parents=True, exist_ok=True)

    handler = FileHandler(logfile_path)
    handler.setFormatter(Formatter(logformat))

    logger = getLogger('aa')
    logger.addHandler(handler)


def set_logger():
    logger = getLogger('aa')
    logformat = '%(asctime)s %(levelname)s %(message)s'

    handler_out = StreamHandler(sys.stdout)
    handler_out.setLevel(INFO)
    handler_out.setFormatter(Formatter(logformat))

    logger.setLevel(INFO)
    logger.addHandler(handler_out)


def get_csv_folds(path, d, use_all=False):
    df = pd.read_csv(path, index_col=0)
    if use_all:
        train = [range(len(df))]
        test = [[]]
    else:
        m = df.max()[0] + 1
        train = [[] for i in range(m)]
        test = [[] for i in range(m)]

        folds = {}
        for i in range(m):
            fold_ids = list(df[df['fold'].isin([i])].index)
            folds.update({i: [n for n, l in enumerate(d) if l in fold_ids]})

        for k, v in folds.items():
            for i in range(m):
                if i != k:
                    train[i].extend(v)
            test[k] = v

    return list(zip(np.array(train), np.array(test)))
