# train
JSON=$1
python 00_gen_folds.py $JSON

for IDX in {2..4}
do
    python 01_train.py $JSON --fold=${IDX};
done

