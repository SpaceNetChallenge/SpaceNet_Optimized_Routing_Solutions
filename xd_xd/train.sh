#!/bin/bash

# Usage:
#     $ docker run -rm -v <local_data_path>:/data:ro \
#           -v <local_writable_area_path>:/wdata \
#           -it XD_XD
#     # ./train.sh /data/train/AOI_2_Vegas_Train \
#                  /data/train/AOI_3_Paris_Train \
#                  /data/train/AOI_4_Shanghai_Train \
#                  /data/train/AOI_5_Khartoum_Train \
#                  /data/train/AOI_7_Moscow_Train \
#                  /data/train/AOI_8_Mumbai_Train

# -------------------------------------------------

# 8bit images & multi-class road masks
# (58min)
# * /wdata/input/train/images_8bit_base/PS-RGB/
# * /wdata/input/train/masks_base/
bash ./preprocessing_train.sh $*

# Generate cv splits
PYTHONPATH=. python aa/cli/sp5r2/gen_folds.py

# train models
# (24h + 19h)
for fold in {0..3}; do
    CMD="bash ./train_gpu.sh ${fold}"
    echo ">>> ${CMD}"
    tmux new-session -d -s fold${fold} ${CMD}
    echo "sleep 30"
    sleep 30
done

while tmux has-session -t fold0; do sleep 1; done
while tmux has-session -t fold1; do sleep 1; done
while tmux has-session -t fold2; do sleep 1; done
while tmux has-session -t fold3; do sleep 1; done

echo "DONE!"
