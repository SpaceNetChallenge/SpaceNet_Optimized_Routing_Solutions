# xdxd's solution

## Setup a docker host

See `setup_aws_p3.8xlarge.sh`.

## Usage

For testing:

```
ubuntu@ip-172-31-23-74:~$ docker build -t xd_xd .
ubuntu@ip-172-31-23-74:~$ docker run \
    --ipc=host \
    --shm-size 4G \
    --gpus all \
    -it \
    --rm \
    -v /home/ubuntu/data:/data:ro \
    -v /home/ubuntu/wdata:/wdata \
    xd_xd bash

(base) root@50d5a398467a:~# ./test.sh \
    /data/test_public/AOI_7_Moscow_Test_public \
    /data/test_public/AOI_8_Mumbai_Test_public \
    /data/test_public/AOI_9_San_Juan_Test_public \
    solution.csv
```

For training:

```
ubuntu@ip-172-31-23-74:~$ docker build -t xd_xd .
ubuntu@ip-172-31-23-74:~$ docker run \
    --ipc=host \
    --shm-size 4G \
    --gpus all \
    -it \
    --rm \
    -v /home/ubuntu/data:/data:ro \
    -v /home/ubuntu/wdata:/wdata \
    xd_xd bash

(base) root@50d5a398467a:~# ./train.sh \
    /data/train/AOI_2_Vegas_Train \
    /data/train/AOI_3_Paris_Train \
    /data/train/AOI_4_Shanghai_Train \
    /data/train/AOI_5_Khartoum_Train \
    /data/train/AOI_7_Moscow_Train \
    /data/train/AOI_8_Mumbai_Train
```
