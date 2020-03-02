#!/bin/bash

TRAIN_DIR=$1  # e.g.) /data/train/AOI_2_Vegas_Train
DATASPLIT_NAME=$2  # "train", "test_public" or "test_private"

mkdir -p /wdata/input/${DATASPLIT_NAME}/images_8bit_base/PS-RGB
PYTHONPATH=. python aa/cli/sp5r2/create_8bit_images.py \
    --indir ${TRAIN_DIR}/PS-MS \
    --outdir /wdata/input/${DATASPLIT_NAME}/images_8bit_base/PS-RGB \
    --rescale_type=perc \
    --percentiles=2,98 \
    --band_order=5,3,2

if [ $DATASPLIT_NAME == "train" ]; then
    mkdir -p /wdata/input/${DATASPLIT_NAME}/masks_base
    PYTHONPATH=. python aa/cli/sp5r2/speed_masks.py \
        --geojson_dir ${TRAIN_DIR}/geojson_roads_speed \
        --image_dir ${TRAIN_DIR}/PS-MS \
        --output_conversion_csv /wdata/input/${DATASPLIT_NAME}/masks_base/roads_train_speed_conversion_binned.csv \
        --output_mask_dir /wdata/input/${DATASPLIT_NAME}/masks_base/train_mask_binned \
        --output_mask_multidim_dir /wdata/input/${DATASPLIT_NAME}/masks_base/train_mask_binned_mc \
        --buffer_distance_meters 2
fi
