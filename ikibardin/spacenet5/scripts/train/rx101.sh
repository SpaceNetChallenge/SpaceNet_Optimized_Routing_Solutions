#!/usr/bin/env bash

set -e


for FOLD_ID in 0 1
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train \
        --config configs/train/rx101_fixed.yml \
        --paths configs/paths.yml \
        --fold ${FOLD_ID};
done