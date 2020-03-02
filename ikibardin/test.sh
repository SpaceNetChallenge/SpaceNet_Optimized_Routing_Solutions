#!/usr/bin/env bash

set -e

ARGLEN=$(($#-1))
ARGS="${@:1:${ARGLEN}}"
LAST="${!#}"


# Convert images
pushd cresi-spacenet
bash scripts/prepare_test_images.sh ${ARGS}
popd

# Copy weights to wdata
mkdir -p /wdata/learning_dumps/spacenet5 && cp -r final_dumps/* /wdata/learning_dumps/spacenet5  # FIXME && rm -r /wdata/learning_dumps/*
mkdir -p /wdata/cresi_results/weights/fpn_fixed && cp -r cresi_weights/* /wdata/cresi_results/weights/fpn_fixed

# Run inference for binary masks
pushd spacenet5
bash scripts/run_binary_inference.sh
popd

# Run inference for speed masks
pushd cresi-spacenet
pushd cresi
python 02_eval.py jsons/fpn_fixed.json --fold 0
popd
popd

# Extract graph
bash extract_graph.sh ${LAST}


