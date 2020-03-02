#!/usr/bin/env bash

set -e

FOLDS_FOLDER=$1
#FPN_MASKS_FOLDER="/wdata/cresi_data/cresi_train/train_mask_binned_mc"
CONVERTED_PATH=$2

FOLDS_NUM=$3

mkdir -p ${CONVERTED_PATH}

python -m src.processing.merge_folds_fpn_cresi \
    --folds-path ${FOLDS_FOLDER} \
    --out-path ${CONVERTED_PATH} \
    --folds-num ${FOLDS_NUM}