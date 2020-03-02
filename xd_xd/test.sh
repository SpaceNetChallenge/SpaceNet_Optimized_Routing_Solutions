#!/bin/bash

# TEST
touch /wdata/TEST_WORKING_DIRECTORY_IS_WRITABLE

# COPY TRAINED MODELS
cp -r /root/working /wdata/
mkdir -p /wdata/input/train/masks_base
cp /root/roads_train_speed_conversion_binned.csv /wdata/input/train/masks_base/roads_train_speed_conversion_binned.csv

# Parse args
TEST_FOLDERS=${@:1:($#-1)}
OUTPUT_PATH=`eval echo '$'{$#}`

# Usage:
#     $ docker run -rm -v <local_data_path>:/data:ro \
#           -v <local_writable_area_path>:/wdata \
#           -it XD_XD
#     # ./test.sh /data/test_public/AOI_7_Moscow_Test_public \
#                 /data/test_public/AOI_8_Mumbai_Test_public \
#                 /data/test_public/AOI_9_San_Juan_Test_public \
#                 solution.csv

# -------------------------------------------------

# 8bit images & multi-class road masks
# (4min)
# OUTPUT:
# * /wdata/input/test_public/images_8bit_base/PS-RGB/

time bash ./preprocessing_test.sh $TEST_FOLDERS

# test models (4x2 models)
# (4min + 9min) = 13min
for fold in {0..3}; do
    CMD="bash ./test_gpu.sh ${fold}"
    echo ">>> ${CMD}"
    tmux new-session -d -s fold${fold} ${CMD}
    echo "sleep 30"
    sleep 30
done

while tmux has-session -t fold0; do sleep 1; done
while tmux has-session -t fold1; do sleep 1; done
while tmux has-session -t fold2; do sleep 1; done
while tmux has-session -t fold3; do sleep 1; done

# --------- IF ensemble
# 13m30.093s
time CUDA_VISIBLE_DEVICES=$FOLD_ID python -W ignore aa/cli/sp5r2/ens.py ensemble \
    -c configs/sn5r2/ens/ens_r50a_serx50.py

time python -W ignore aa/cli/sp5r2/clean_graph.py \
    --input-file /wdata/working/sp5r2/models/solution/ens_r50a_serx50/solution_debug.csv \
    --output-file /wdata/working/sp5r2/models/solution/ens_r50a_serx50/solution_postprocessed.csv \
	--args $TEST_FOLDERS

time CUDA_VISIBLE_DEVICES=$FOLD_ID python -W ignore aa/cli/sp5r2/ens.py remove-small-edges \
    --input-path /wdata/working/sp5r2/models/solution/ens_r50a_serx50/solution_postprocessed.csv \
    --output-path $OUTPUT_PATH

echo "DONE!"
