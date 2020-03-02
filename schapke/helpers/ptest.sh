set -e

#mkdir -p /wdata
#rm -rf /results/results
#rm -rf /wdata/test

### Container files
cd /work

#### PUBLIC ####
python data_prep/create_8bit_images.py \
    --indirs \
        /spacenet/dataset/AOI_9_San_Juan/ \
        /spacenet/dataset/test_AOI_7_Moscow/ \
        /spacenet/dataset/test_AOI_8_Mumbai/ \
    --outdir /wdata/test/8bit \
    --band_order 1,2,3

python data_prep/create_psms.py \
    --indirs \
        /spacenet/dataset/AOI_9_San_Juan/ \
        /spacenet/dataset/test_AOI_7_Moscow/ \
        /spacenet/dataset/test_AOI_8_Mumbai/ \
    --outdir /wdata/test/psms
#### --- ####

#### PRIVATE ####
#python data_prep/create_8bit_images.py \
#    --indirs \
#        /spacenet/dataset/test/AOI_2_Vegas/ \
#        /spacenet/dataset/test/AOI_3_Paris/ \
#        /spacenet/dataset/test/AOI_4_Shanghai/ \
#        /spacenet/dataset/test/AOI_5_Khartoum/ \
#        /spacenet/dataset/test/AOI_7_Moscow/ \
#        /spacenet/dataset/test/AOI_8_Mumbai/ \
#    --outdir /wdata/test/8bit \
#    --band_order 1,2,3
#
#python data_prep/create_psms.py \
#    --indirs \
#        /spacenet/dataset/test/AOI_2_Vegas/ \
#        /spacenet/dataset/test/AOI_3_Paris/ \
#        /spacenet/dataset/test/AOI_4_Shanghai/ \
#        /spacenet/dataset/test/AOI_5_Khartoum/ \
#        /spacenet/dataset/test/AOI_7_Moscow/ \
#        /spacenet/dataset/test/AOI_8_Mumbai/ \
#    --outdir /wdata/test/psms 
#### --- ####

#### SMALL TEST ####
#python data_prep/create_8bit_images.py \
#    --indirs \
#        /spacenet/dataset/smallTrain/AOI_7_Moscow/ \
#    --outdir /wdata/test/8bit \
#    --band_order 1,2,3
#
#python data_prep/create_psms.py \
#    --indirs \
#        /spacenet/dataset/smallTrain/AOI_7_Moscow/ \
#    --outdir /wdata/test/psms 
#### --- ####


#python 02_eval.py jsons/10_focal_sdice_PSMS.json
#python 02_eval.py jsons/10_focal_sdice.json
#python 02_eval.py jsons/10_bce_dice.json

python 02_eval.py jsons/8_bce_dice.json
#python 02_eval.py jsons/8_bce_dice2.json
python 02_eval.py jsons/8_focal_sdice.json
#python 02_eval.py jsons/8_focal_sdice2.json

#python 03a_merge_preds.py jsons/10_focal_sdice.json
python 03a_merge_preds.py jsons/8_focal_sdice.json

##python 04_skeletonize.py jsons/10_focal_sdice.json
python 04_skeletonize.py jsons/8_focal_sdice.json
#
###python 05_wkt_to_G.py jsons/10_focal_sdice.json
python 05_wkt_to_G.py jsons/8_focal_sdice.json
#
###python 06_infer_speed.py jsons/10_focal_sdice.json
python 06_infer_speed.py jsons/8_focal_sdice.json
#
###python 07a_create_submission_wkt.py jsons/10_focal_sdice.json /submission.csv
python 07a_create_submission_wkt.py jsons/8_focal_sdice.json /submission.csv
