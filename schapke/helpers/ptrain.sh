set -e
cd /work

rm -rf /wdata/train
rm -rf /results/results/folds4.csv

mkdir -p /wdata/train
mkdir -p /results/results

#### Full Dataset ####
#python data_prep/speed_masks.py \
#    --indirs \
#        /spacenet/dataset/train/AOI_5_Khartoum \
#        /spacenet/dataset/train/AOI_4_Shanghai/ \
#        /spacenet/dataset/train/AOI_3_Paris/ \
#        /spacenet/dataset/train/AOI_2_Vegas/ \
#        /spacenet/dataset/train/AOI_7_Moscow/ \
#        /spacenet/dataset/train/AOI_8_Mumbai/ \
#    --output_conversion_csv /wdata/SN5_roads_train_speed_conversion_binned.csv \
#    --output_mask_dir /wdata/train/masks \
#    --output_mask_multidim_dir /wdata/train/masks_binned \
#    --buffer_distance_meters 2 \
#    --num_classes 8

#python data_prep/create_8bit_images.py \
#    --indirs \
#        /spacenet/dataset/train/AOI_2_Vegas/ \
#        /spacenet/dataset/train/AOI_3_Paris/ \
#        /spacenet/dataset/train/AOI_4_Shanghai/ \
#        /spacenet/dataset/train/AOI_5_Khartoum/ \
#        /spacenet/dataset/train/AOI_7_Moscow/ \
#        /spacenet/dataset/train/AOI_8_Mumbai/ \
#    --outdir /wdata/train/8bit \
#    --rescale_type perc \
#    --percentiles 2,98 \
#    --band_order 5,3,2
#### --- ####

#### SN5 Dataset ####
#python data_prep/speed_masks.py \
#    --indirs \
#        /spacenet/dataset/train/AOI_7_Moscow/ \
#        /spacenet/dataset/train/AOI_8_Mumbai/ \
#    --output_conversion_csv /wdata/SN5_roads_train_speed_conversion_binned.csv \
#    --output_mask_dir /wdata/train/masks \
#    --output_mask_multidim_dir /wdata/train/masks_binned \
#    --buffer_distance_meters 2 \
#    --num_classes 8
#
#python data_prep/create_8bit_images.py \
#    --indirs \
#        /spacenet/dataset/train/AOI_8_Mumbai/ \
#        /spacenet/dataset/train/AOI_7_Moscow/ \
#    --outdir /wdata/train/8bit \
#    --rescale_type perc \
#    --percentiles 2,98 \
#    --band_order 5,3,2
#### --- ####

#### Small Dataset ####
#python data_prep/speed_masks.py \
#    --indirs \
#        /spacenet/dataset/smallTrain/AOI_7_Moscow/ \
#    --output_conversion_csv /wdata/SN5_roads_train_speed_conversion_binned.csv \
#    --output_mask_multidim_dir /wdata/train/masks_binned \
#    --output_mask_dir /wdata/train/masks \
#    --buffer_distance_meters 2 \
#    --num_classes 8
#
#python data_prep/create_8bit_images.py \
#    --indirs \
#        /spacenet/dataset/smallTrain/AOI_7_Moscow \
#    --outdir /wdata/train/8bit \
#    --rescale_type perc \
#    --percentiles 2,98 \
#    --band_order 1,2,3
#### --- ####


#python 00_gen_folds.py jsons/10_focal_sdice.json
python 00_gen_folds.py jsons/8_focal_sdice2.json


#python 01_train.py jsons/8_bce_dice.json --num_workers=0 --fold=0
#python 01_train.py jsons/8_bce_dice2.json --num_workers=0 --fold=1
#python 01_train.py jsons/8_focal_sdice.json --num_workers=0 --fold=2
python 01_train.py jsons/8_focal_sdice2.json --num_workers=0 --fold=3

#python 01_train.py jsons/10_focal_sdice.json     --num_workers=0
#python 01_train.py jsons/10_focal_sdice_PSMS.json --num_workers=0
#python 01_train.py jsons/10_bce_dice.json        --num_workers=0
#python 01_train.py jsons/10_bce_dice_PSMS.json   --num_workers=0
