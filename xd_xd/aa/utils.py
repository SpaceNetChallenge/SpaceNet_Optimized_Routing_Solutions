import os
import importlib

from easydict import EasyDict as edict
import numpy as np
import pandas as pd


def is_development_node():
    return True if os.uname().nodename == 'resona' else False


def load_config(config):
    mod = importlib.import_module(config.rstrip('.py').replace('/', '.'))
    conf = edict(mod.CONFIG)
    return conf


def get_csv_folds(path, d):
    df = pd.read_csv(path, index_col=0)
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
