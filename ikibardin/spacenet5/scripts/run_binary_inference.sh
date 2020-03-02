#!/usr/bin/env bash

set -e

# Prepare .csv with image ids
bash scripts/processing/make_test_csv.sh

# Predict with binary models
bash scripts/predict/rx101.sh
bash scripts/predict/srx50.sh
bash scripts/predict/rn50_fpn.sh


# Blend masks
bash scripts/predict/blend.sh
