#!/usr/bin/env bash

set -e

for FOLD_ID in 2 3
do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train \
        --config configs/predict/rn50_fpn.yml \
        --paths configs/paths.yml \
        --no-train \
        --fold ${FOLD_ID};
done