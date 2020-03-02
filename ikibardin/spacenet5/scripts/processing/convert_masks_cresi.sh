#!/usr/bin/env bash

set -e

MASKS_FOLDER=$1
CONVERTED_PATH=$2

mkdir -p "/wdata/cresi_data"
mkdir -p ${CONVERTED_PATH}

python -m src.processing.convert_masks_cresi \
    --converted-path ${CONVERTED_PATH} \
    --masks-path ${MASKS_FOLDER}