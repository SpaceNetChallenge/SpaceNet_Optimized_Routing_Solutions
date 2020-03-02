#!/usr/bin/env bash

set -e

BINARY_MASKS=$1
SPEED_MASKS=$2
CONVERTED_PATH=$3

mkdir -p ${CONVERTED_PATH}

python -m src.processing.merge_masks_cresi \
    --unet-path ${BINARY_MASKS} \
    --fpn-path ${SPEED_MASKS} \
    --out-path ${CONVERTED_PATH}