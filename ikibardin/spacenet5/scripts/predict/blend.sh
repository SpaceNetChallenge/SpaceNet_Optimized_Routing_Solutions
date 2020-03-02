#!/usr/bin/env bash

set -e

DUMPS="/wdata/learning_dumps/spacenet5"

OUTPUT="/wdata/binary_masks"

python -m src.processing.blend \
    --dumps ${DUMPS} \
    --output ${OUTPUT};
