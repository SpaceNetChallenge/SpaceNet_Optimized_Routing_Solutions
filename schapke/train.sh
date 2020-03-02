set -e 

# rm -rf /wdata
# rm -rf /results

mkdir -p /wdata/train
mkdir -p /results/results

cd /work

python data_prep/speed_masks.py \
    --indirs "$@" \
    --output_conversion_csv /wdata/SN5_roads_train_speed_conversion_binned.csv \
    --output_mask_dir /wdata/train/masks \
    --output_mask_multidim_dir /wdata/train/masks_binned \
    --buffer_distance_meters 2 \
    --num_classes 8

python data_prep/create_8bit_images.py \
    --indir "$@" \
    --outdir /wdata/train/8bit \
    --rescale_type perc \
    --percentiles 2,98 \
    --band_order 5,3,2

#python data_prep/create_psms.py \
#    --indirs "$@" \
#    --outdir /wdata/test/psms 

python 00_gen_folds.py jsons/8_focal_sdice.json

mkdir -p /wdata/out

CUDA_VISIBLE_DEVICES="0" python 01_train.py jsons/8_focal_sdice.json       --num_workers=0 --fold=0 \
    &> /wdata/out/focal_sdice.out &
CUDA_VISIBLE_DEVICES="1" python 01_train.py jsons/8_focal_sdice2.json      --num_workers=0 --fold=1 \
    &> /wdata/out/focal_sdice2.out &
CUDA_VISIBLE_DEVICES="2" python 01_train.py jsons/8_bce_dice.json          --num_workers=0 --fold=2 \
    &> /wdata/out/bce_dice.out &
CUDA_VISIBLE_DEVICES="3" python 01_train.py jsons/8_bce_dice2.json         --num_workers=0 --fold=3 \
    &> /wdata/out/bce_dice2.out &

wait
echo "All Trained"
