#!/usr/bin/env bash

set -e

ARGS="${@}"

OUTPUT_BINNED="/wdata/cresi_data/cresi_train/train_mask_binned"
OUTPUT_MULTIDIM="/wdata/cresi_data/cresi_train/train_mask_binned_mc"

mkdir -p ${OUTPUT_BINNED}
mkdir -p ${OUTPUT_MULTIDIM}


# SN5 - binned
for DIR in ${ARGS}
do
    python cresi/data_prep/speed_masks.py \
      --geojson_dir="${DIR}/geojson_roads_speed" \
      --image_dir="${DIR}/PS-MS" \
      --output_conversion_csv=/wdata/cresi_data/cresi_train/SN5_roads_train_speed_conversion_binned.csv \
      --output_mask_dir=${OUTPUT_BINNED} \
      --output_mask_multidim_dir=${OUTPUT_MULTIDIM} \
      --buffer_distance_meters=2;
done