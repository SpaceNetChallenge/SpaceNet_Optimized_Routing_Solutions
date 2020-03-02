#!/usr/bin/env bash

set -e

MASKS="/wdata/cresi_data/cresi_train/train_mask_binned_mc"

OUTPUT_CSV="tables/folds_v4.csv"


mkdir -p "tables"

python -m src.processing.split_folds \
    --masks   ${MASKS} \
    --out-csv ${OUTPUT_CSV};
