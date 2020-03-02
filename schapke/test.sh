set -e

rm -rf /results/results
rm -rf /wdata/test
mkdir -p /wdata
mkdir -p results/results

cd /work
python data_prep/create_8bit_images.py \
    --indirs "${@:1:$#-1}" \
    --outdir /wdata/test/8bit \
    --band_order 5,3,2

#python data_prep/create_psms.py \
#    --indirs "$@" \
#    --outdir /wdata/test/psms 


CUDA_VISIBLE_DEVICES="0" python 02_eval.py jsons/8_bce_dice.json \
    &> /wdata/test_bce_dice.out &

CUDA_VISIBLE_DEVICES="1" python 02_eval.py jsons/8_bce_dice2.json \
    &> /wdata/test_bce_dice2.out &

CUDA_VISIBLE_DEVICES="2" python 02_eval.py jsons/8_focal_sdice.json \
    &> /wdata/test_focal_sdice.out &

CUDA_VISIBLE_DEVICES="3" python 02_eval.py jsons/8_focal_sdice2.json \
    &> /wdata/test_focal_sdice2.out &
wait

#python 02_eval.py jsons/8_bce_dice.json \
#    &> /wdata/test_bce_dice.out
#
#python 02_eval.py jsons/8_bce_dice2.json \
#    &> /wdata/test_bce_dice2.out
#
#python 02_eval.py jsons/8_focal_sdice.json \
#    &> /wdata/test_focal_sdice.out
#
#python 02_eval.py jsons/8_focal_sdice2.json \
#    &> /wdata/test_focal_sdice2.out


python 03a_merge_preds.py jsons/8_focal_sdice.json

python 04_skeletonize.py jsons/8_focal_sdice.json

python 05_wkt_to_G.py jsons/8_focal_sdice.json

python 06_infer_speed.py jsons/8_focal_sdice.json

python 07a_create_submission_wkt.py jsons/8_focal_sdice.json "${@:$#}"
