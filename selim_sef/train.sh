#!/usr/bin/env bash

mkdir -p /wdata/PS-RGB
mkdir -p /wdata/PS-MS

for var in "$@"
do
PYTHONPATH=. python data_tools/create_8bit_images.py \
  --indir=$var/PS-MS \
  --outdir=/wdata/PS-RGB \
  --rescale_type=perc \
  --percentiles=2,98 \
  --band_order=5,3,2
PYTHONPATH=. python data_tools/create_8bit_images.py \
  --indir=$var/PS-MS \
  --outdir=/wdata/PS-MS \
  --rescale_type=perc \
  --percentiles=2,98 \
  --band_order=5,3,2,1,4,6,7,8

PYTHONPATH=. python data_tools/speed_masks.py \
  --geojson_dir=$var/geojson_roads_speed \
  --image_dir=$var/PS-MS \
  --output_conversion_csv=/wdata/SN3_roads_train_speed_conversion_binned10.csv \
  --output_mask_dir=/wdata/train_mask_binned \
  --output_mask_multidim_dir=/wdata/train_mask_binned_mc_10 \
  --buffer_distance_meters=2
done

PYTHONPATH=. python data_tools/generate_spacenet_dataset.py  --data-dirs "$@" --out-dir /wdata


mkdir -p /wdata/logs
mkdir -p weights

timeout 24h python train.py --gpu 0 --test_every 6 --folds-csv folds4.csv --fold 0 --config configs/irv2.json --freeze-epochs 1 > /wdata/logs/ir0 &
timeout 24h python train.py --gpu 1 --test_every 6 --folds-csv folds4.csv --fold 1 --config configs/irv2.json --freeze-epochs 1 > /wdata/logs/ir1 &
timeout 24h python train.py --gpu 2 --test_every 6 --folds-csv folds4.csv --fold 2 --config configs/dpn92mc.json --freeze-epochs 0 > /wdata/logs/d2 &
timeout 24h python train.py --gpu 3 --test_every 6 --folds-csv folds4.csv --fold 3 --config configs/d92.json --freeze-epochs 1 > /wdata/logs/d3 &

wait

timeout 6h python -u -m torch.distributed.launch --nproc_per_node=4 train.py --workers 24 --distributed --config configs/irv2_tune.json --test_every 1 --folds-csv folds4.csv --fold 0  \
 --freeze-epochs 0 --from-zero --resume weights/spacenet_irv_unet_inceptionresnetv2_0_last > /wdata/logs/ir0
timeout 6h python -u -m torch.distributed.launch --nproc_per_node=4 train.py --workers 24 --distributed --config configs/dpn92mc_tune.json --test_every 1 --folds-csv folds4.csv --fold 2  \
 --freeze-epochs 0 --from-zero --resume weights/spacenet_dpn_unet_mc_dpn92_mc_2_last > /wdata/logs/d2
timeout 6h python -u -m torch.distributed.launch --nproc_per_node=4 train.py --workers 24 --distributed --config configs/irv2_tune.json --test_every 1 --folds-csv folds4.csv --fold 1  \
 --freeze-epochs 0 --from-zero --resume weights/spacenet_irv_unet_inceptionresnetv2_1_last > /wdata/logs/ir1
timeout 6h python -u -m torch.distributed.launch --nproc_per_node=4 train.py --workers 24 --distributed --config configs/d92_tune.json --test_every 1 --folds-csv folds4.csv --fold 3  \
 --freeze-epochs 0 --from-zero --resume weights/spacenet_dpn_unet_dpn92_3_last > /wdata/logs/d3


