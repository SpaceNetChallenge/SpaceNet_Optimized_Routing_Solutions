#!/usr/bin/env bash

set -e

MASKS_FOLDER="/wdata/cresi_results/sn5_baseline/folds"

SAVE_PATH="/solution/cresi_solution.csv"

mkdir -p "tables"
mkdir -p "/solution"

python -m src.tools.vectorize_tools \
    --masks-folder ${MASKS_FOLDER} \
    --save-path ${SAVE_PATH};
