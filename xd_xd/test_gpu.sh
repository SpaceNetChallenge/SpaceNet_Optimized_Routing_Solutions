#!/bin/bash
FOLD_ID=$1

# 3m50s
# OUTPUT:
# * /wdata/working/sp5r2/models/preds/r34a/fold0_test/fold0_SN5_roads_test_public_AOI_...
CUDA_VISIBLE_DEVICES=$FOLD_ID python -W ignore aa/cli/sp5r2/inference.py evaltest \
    -c configs/sn5r2/r50a.py \
    -f $FOLD_ID

# xx min
# OUTPUT:
# * /wdata/working/sp5r2/models/preds/x50b/fold0_test/fold0_SN5_roads_test_public_AOI_...
CUDA_VISIBLE_DEVICES=$FOLD_ID python -W ignore aa/cli/sp5r2/inference.py evaltest \
    -c configs/sn5r2/serx50_focal.py \
    -f $FOLD_ID
