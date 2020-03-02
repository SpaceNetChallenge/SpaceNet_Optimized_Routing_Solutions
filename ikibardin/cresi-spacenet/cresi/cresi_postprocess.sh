#!/usr/bin/env bash

set -e

JSON=$1
OUTPUT_CSV=$2

#python 03b_stitch.py $JSON
python 04_skeletonize.py ${JSON}
python 05_wkt_to_G.py ${JSON}
python 06_infer_speed.py ${JSON}
python 07a_create_submission_wkt.py \
    --config-path ${JSON} \
    --out-csv ${OUTPUT_CSV}