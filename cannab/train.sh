mkdir -p foo /wdata/logs

echo "Training with folders:"
echo $@

#echo "Prerocessing and creating masks..."
#nohup python create_masks.py "$@" > /wdata/logs/create_masks.out &
#wait
# Time: 119.551 min

#
#echo "Resizing to 960*960"
#nohup python create_masks_960.py "$@" > /wdata/logs/create_masks_960.out &
#wait
#echo "Data preprocessed"
# Time: 47.633 min

#echo "training dpn92 on 960*960"
#nohup python train92_9ch_960.py 3 0,1 > /wdata/logs/train92_9ch_960_3.out &
#nohup python train92_9ch_960.py 4 2,3 > /wdata/logs/train92_9ch_960_4.out &
#wait
# Time: 1241.225 min

#echo "tuning dpn92 on full"
#nohup python tune92_9ch.py 3 0,1 > /wdata/logs/tune92_9ch_3.out &
#nohup python tune92_9ch.py 4 2,3 > /wdata/logs/tune92_9ch_4.out &
#wait
## loaded checkpoint 'dpn92_9ch_3_0_best' (epoch 39, best_score 0.6260168007132588)
## Time: 576.933 min

# echo "training se50 on full"
# nohup python train50_9ch.py 5 0,1 > /wdata/logs/train50_9ch_5.out &
# nohup python train50_9ch.py 5 0,1,2,3 > /wdata/logs/train50_9ch_5.out &
# nohup python train50_9ch.py 6 2,3 > /wdata/logs/train50_9ch_6.out &
# wait
# Time: 524.594 min

#echo "tuning se50 on 960*960"
#nohup python tune50_9ch_960.py 5 0,1 > /wdata/logs/tune50_9ch_960_5.out &
#nohup python tune50_9ch_960.py 6 2,3 > /wdata/logs/tune50_9ch_960_6.out &
#wait
## Time: 274.455 min

#
#echo "training res34 on 960*960"
#nohup python train34_9ch_960.py 1 0,1 > /wdata/logs/train34_9ch_960_1.out &
#nohup python train34_9ch_960.py 2 2,3 > /wdata/logs/train34_9ch_960_2.out &
#wait
### Time: 660 min
#
#echo "training res34 on full"
#nohup python train34_9ch_full.py 7 0,1 > /wdata/logs/train34_9ch_full_7.out &
#nohup python train34_9ch_full.py 8 2,3 > /wdata/logs/train34_9ch_full_8.out &
#wait
## time # 440 min

echo "All models trained!"

# total GPU time 3764 minutes = 63 hours with all four gpus
