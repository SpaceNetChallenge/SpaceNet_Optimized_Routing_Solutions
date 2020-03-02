#!/bin/bash

for TRAIN_DIR in $*; do
    echo bash ./preprocessing_.sh ${TRAIN_DIR} "train"
    bash ./preprocessing_.sh ${TRAIN_DIR} "train"
done
