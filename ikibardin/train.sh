#!/usr/bin/env bash

set -e

ARGS="${@}"

# Remove weights shipped with the container
rm -rf final_dumps
rm -rf cresi_weights

# Prepare image for training
pushd cresi-spacenet
bash scripts/prepare_images.sh ${ARGS}
popd

# Prepare masks for training
pushd cresi-spacenet
bash scripts/prepare_masks.sh ${ARGS}
popd

# Split data into folds
pushd spacenet5
bash scripts/processing/split_folds.sh
popd

# Train binary models
pushd spacenet5
bash scripts/train/rx101.sh
bash scripts/train/srx50.sh
bash scripts/train/rn50_fpn.sh
popd

# Train speed model
pushd cresi-spacenet
pushd cresi
python 00_gen_folds.py jsons/fpn_fixed.json
python 01_train.py jsons/fpn_fixed.json --fold=0
popd
popd

# Save trained weights to use them during inference
cp -r /wdata/learning_dumps/spacenet5 final_dumps
cp -r /wdata/cresi_results/weights/fpn_fixed cresi_weights
