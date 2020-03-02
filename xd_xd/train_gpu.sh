#!/bin/bash
FOLD_ID=$1

CUDA_VISIBLE_DEVICES=$FOLD_ID python -W ignore aa/cli/sp5r2/train.py \
    -c configs/sn5r2/r50a.py \
    -f $FOLD_ID

CUDA_VISIBLE_DEVICES=$FOLD_ID python -W ignore aa/cli/sp5r2/train.py \
    -c configs/sn5r2/serx50_focal.py \
    -f $FOLD_ID
