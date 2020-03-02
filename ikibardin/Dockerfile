FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install --fix-missing -y --no-install-recommends \
    build-essential \
    apt-transport-https \
    ca-certificates \
    cmake \
    curl \
    git \
    software-properties-common \
    wget \
    unzip \
    nano


RUN git clone https://github.com/ikibardin/libspatialindex \
    && cd libspatialindex \
    && cmake . -DCMAKE_INSTALL_PREFIX=/usr \
		    -DCMAKE_SYSTEM_NAME=Linux \
			-DCMAKE_BUILD_TYPE=Release \
			&& make && make install


RUN pip install --upgrade pip

RUN apt-get update \
    && add-apt-repository -y ppa:ubuntugis/ppa \
    && apt-get update \
    && apt-get install -y gdal-bin libgdal-dev python-gdal python3-gdal \
    && export CPLUS_INCLUDE_PATH=/usr/include/gdal \
    && export C_INCLUDE_PATH=/usr/include/gdal \
    && pip install GDAL==$(gdal-config --version | awk -F'[.]' '{print $1"."$2}')

RUN git clone https://github.com/NVIDIA/apex /tmp/apex \
    && pip install -v --no-cache-dir /tmp/apex \
    && rm -r /tmp/apex

COPY power-fist-segmentation /code/power-fist-segmentation
RUN pip install -e /code/power-fist-segmentation

COPY requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt --no-cache-dir


RUN curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1MCYgAbZznUkW_tNaHpJ5QGLtYDgurb80" > /dev/null \
    && curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1MCYgAbZznUkW_tNaHpJ5QGLtYDgurb80" -o "/code/final_dumps.zip" \
    && unzip /code/final_dumps.zip -d /code


RUN curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1AIHVE6J7D1gMwbzwCZoMWUM_ATGhtX1C" > /dev/null \
    && curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1AIHVE6J7D1gMwbzwCZoMWUM_ATGhtX1C" -o "/code/cresi_weights.zip" \
    && unzip /code/cresi_weights.zip -d /code

RUN mkdir -p /root/.cache/torch/checkpoints

RUN wget https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth -O /root/.cache/torch/checkpoints/ig_resnext101_32x8-c38310e5.pth

RUN wget http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth -O /root/.cache/torch/checkpoints/se_resnext50_32x4d-a260b3a4.pth

RUN wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O /root/.cache/torch/checkpoints/resnet50-19c8e357.pth

COPY . /code
WORKDIR /code
