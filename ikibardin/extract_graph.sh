#!/usr/bin/env bash

set -e

BINARY_MASKS="/wdata/binary_masks"

SPEED_MASKS="/wdata/cresi_results/fpn_fixed/folds"  # I suppose that they are already merged and in cresi format.


MERGED_BINARY_SPEED_MASKS="/wdata/cresi_results/fpn_fixed_unet/folds"

JSON_CONFIG="/code/cresi-spacenet/cresi/jsons/predict_baseline.json"

OUTPUT_CSV=$1


# Merge with speed masks.
pushd spacenet5
bash scripts/processing/merge_masks_cresi.sh ${BINARY_MASKS} ${SPEED_MASKS} ${MERGED_BINARY_SPEED_MASKS}
popd


# Multichannel postprocessing from cresi. (Changed)
pushd cresi-spacenet
pushd cresi
bash cresi_postprocess.sh ${JSON_CONFIG} ${OUTPUT_CSV}
popd
popd
