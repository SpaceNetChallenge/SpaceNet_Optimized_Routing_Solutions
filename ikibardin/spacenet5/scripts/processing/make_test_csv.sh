#!/usr/bin/env bash

set -e

OUTPUT_CSV="tables/test.csv"
PATHS_CONFIG="configs/paths.yml"

mkdir -p "tables"

python -m src.processing.make_test_csv \
    --out-csv ${OUTPUT_CSV} \
    --paths ${PATHS_CONFIG};