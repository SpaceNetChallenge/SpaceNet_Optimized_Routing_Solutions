#mkdir -p foo /wdata/logs
#
#echo "Preparing test files..."
#rm /wdata/test_png -r -f
#rm /wdata/test_png_5_3_0 -r -f
#rm /wdata/test_png_pan_6_7 -r -f
#nohup python prepare_test.py "$@" > /wdata/logs/prepare_test.out &
## Time: 17.820 min for AOI 7,8,9
#wait
#
#rm /wdata/test_pred -r -f
#echo "Predicting Full images"
#nohup python predict.py > /wdata/logs/predict.out &
## Time: 76.945 min
#wait
#
#rm /wdata/test_pred_960 -r -f
#echo "Predicting 960*960 images"
#nohup python predict960.py > /wdata/logs/predict960.out &
## Time: 45.379 min
#wait
#
#echo "Creating length submission"
#nohup python create_submission.py > /wdata/logs/create_submission.out &
## Submission file created! Time: 21.811 min
#wait

echo "Creating speed submission"
nohup python create_submission_speed.py "$@" > /wdata/logs/create_submission_speed.out &
wait
echo "Submission created!"
# real    0m40.830s

# total 145 minutes + 18 for preprocessing 
