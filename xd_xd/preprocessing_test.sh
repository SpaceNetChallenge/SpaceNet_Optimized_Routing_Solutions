#!/bin/bash

# Usage: bash preprocessing.sh [<data_folder>] ...

# Create 8bit images
for TEST_DIR in $*; do
    bash ./preprocessing_.sh ${TEST_DIR} "test_public"
done
