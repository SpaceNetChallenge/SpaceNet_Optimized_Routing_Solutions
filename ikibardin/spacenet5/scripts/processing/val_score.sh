#!/usr/bin/env bash

set -e

TRUTH_DIR="/wdata/train_geojsons" #@FIXME add correct path
VAL_CSV="/solution/val_solution.csv" #@FIXME add correct path
IM_DIR="/wdata/cresi_data/8bit/PS-RGB" #@FIXME add correct path

python -m src.apls.apls \
    --test_method gt_json_prop_wkt \
    --truth_dir ${TRUTH_DIR} \
    --im_dir ${IM_DIR} \
    --prop_wkt_file ${VAL_CSV};
