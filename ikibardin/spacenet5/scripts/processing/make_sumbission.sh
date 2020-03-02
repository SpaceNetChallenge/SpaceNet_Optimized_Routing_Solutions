#!/usr/bin/env bash

set -e

MASKS_FOLDER="/wdata/learning_dumps/spacenet5/srx50_hw_0/fold_0/predictions/fold0_stage4_epoch17_metric0.50561/test"

SAVE_PATH="/solution/solution.csv"

mkdir -p "tables"
mkdir -p "/solution"

python -m src.tools.vectorize_tools \
    --masks-folder ${MASKS_FOLDER} \
    --save-path ${SAVE_PATH};
