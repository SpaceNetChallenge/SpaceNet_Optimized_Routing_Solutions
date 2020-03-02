#!/usr/bin/env bash

all_args=( $@ )
arg_len=${#all_args[@]}
out_file=${all_args[$arg_len-1]}
city_dirs=${all_args[@]:0:$arg_len-1}


mkdir -p /wdata/test-PS-RGB
mkdir -p /wdata/test-PS-MS

for var in $city_dirs
do
echo "$var"
PYTHONPATH=. python data_tools/create_8bit_images.py \
  --indir=$var/PS-MS \
  --outdir=/wdata/test-PS-RGB \
  --rescale_type=perc \
  --percentiles=2,98 \
  --band_order=5,3,2

PYTHONPATH=. python data_tools/create_8bit_images.py \
  --indir=$var/PS-MS \
  --outdir=/wdata/test-PS-MS \
  --rescale_type=perc \
  --percentiles=2,98 \
  --band_order=5,3,2,1,4,6,7,8
done

mkdir -p /wdata/results
python predict.py --gpu 0 --config configs/irv2.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/irv_d1 --model weights/spacenet_irv_unet_inceptionresnetv2_1_best_dice &
python predict.py --gpu 1 --config configs/irv2.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/irv_l1 --model weights/spacenet_irv_unet_inceptionresnetv2_1_last &
python predict.py --gpu 2 --config configs/dpn92mc.json --data-path /wdata/test-PS-MS/ --dir /wdata/results/d92_a2 --model weights/spacenet_dpn_unet_mc_dpn92_mc_2_best_apls &
python predict.py --gpu 3 --config configs/dpn92mc.json --data-path /wdata/test-PS-MS/ --dir /wdata/results/d92_d2 --model weights/spacenet_dpn_unet_mc_dpn92_mc_2_best_dice &
wait
python predict.py --gpu 0 --config configs/dpn92mc.json --data-path /wdata/test-PS-MS/ --dir /wdata/results/d92_l2 --model weights/spacenet_dpn_unet_mc_dpn92_mc_2_last &
python predict.py --gpu 1 --config configs/d92.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/d92_a3 --model weights/spacenet_dpn_unet_dpn92_3_best_apls &
python predict.py --gpu 2 --config configs/d92.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/d92_d3 --model weights/spacenet_dpn_unet_dpn92_3_best_dice &
python predict.py --gpu 3 --config configs/d92.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/d92_l3 --model weights/spacenet_dpn_unet_dpn92_3_last &
wait
python predict.py --gpu 0 --config configs/irv2.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/irv_a0 --model weights/spacenet_irv_unet_inceptionresnetv2_0_best_apls &
python predict.py --gpu 1 --config configs/irv2.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/irv_d0 --model weights/spacenet_irv_unet_inceptionresnetv2_0_best_dice &
python predict.py --gpu 2 --config configs/irv2.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/irv_l0 --model weights/spacenet_irv_unet_inceptionresnetv2_0_last &
python predict.py --gpu 3 --config configs/irv2.json --data-path /wdata/test-PS-RGB/ --dir /wdata/results/irv_a1 --model weights/spacenet_irv_unet_inceptionresnetv2_1_best_apls &
wait

python ensemble.py --ensembling_cpu_threads 28 --ensembling_dir /wdata/results/ensemble --folds_dir /wdata/results  --dirs_to_ensemble irv_a0 irv_d0 irv_l0 irv_a1 irv_d1 irv_l1 d92_a2 d92_d2 d92_l2 d92_a3 d92_d3 d92_l3

python 04_skeletonize.py configs/baseline.json
python 05_wkt_to_G.py configs/baseline.json
python 06_infer_speed.py configs/baseline.json
python 07a_create_submission_wkt.py --root_dir /wdata/results

cp /wdata/results/solution.csv $out_file
