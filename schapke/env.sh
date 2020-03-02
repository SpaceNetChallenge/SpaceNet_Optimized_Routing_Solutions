set -e

docker build -t cresi_schapke .

#docker run --gpus all -it \
#    -v /home/schapke/projects/topcoder/spaceNet/:/spacenet \
#    -v /home/schapke/projects/topcoder/spaceNet/dataset/train/8bit:/wdata/train/8bit \
#    -v /home/schapke/projects/topcoder/spaceNet/dataset/train/masks8:/wdata/train/masks_binned \
#    -v /home/schapke/projects/topcoder/spaceNet/dataset/train/psms:/wdata/train/psms \
#    -v /home/schapke/projects/topcoder/spaceNet/solutions/cresi_schapke/cresi:/work \
#    -v /home/schapke/projects/topcoder/spaceNet/solutions/cresi_schapke/helpers:/helpers \
#    -v /home/schapke/projects/topcoder/spaceNet/solutions/cresi_schapke/results:/results \
#    cresi_schapke


#docker run --gpus all -it \
#    -v /dev/data/8bit:/wdata/train/8bit \
#    -v /dev/data/masks10:/wdata/train/masks_binned \
#    -v /dev/data/spaceNet/results:/results \
#    cresi_schapke

#docker run --gpus all -it \
#    -v /home/schapke/projects/topcoder/spaceNet/dataset/train/8bit:/wdata/train/8bit \
#    -v /home/schapke/projects/topcoder/spaceNet/dataset/train/masks10:/wdata/train/masks_binned \
#    -v /home/schapke/projects/topcoder/spaceNet/:/spacenet \
#    cresi_schapke

docker run --gpus all -it \
    -v /home/schapke/projects/topcoder/spaceNet/:/spacenet \
    cresi_schapke
