FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

# prep apt-get and cudnn
RUN apt-get update && apt-get install -y --no-install-recommends \
	    apt-utils  && \
    rm -rf /var/lib/apt/lists/*

# install requirements
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    apt-utils \
    bc \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libgdal-dev \
    libssl-dev \
    libffi-dev \
    libncurses-dev \
    libgl1 \
    jq \
    nfs-common \
    parallel \
    python-dev \
    python-pip \
    python-wheel \
    python-setuptools \
    unzip \
    vim \
	tmux \
    wget \
    build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/bin:$PATH

# install anaconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV TINI_VERSION v0.16.1
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# use conda-forge instead of default channel
RUN conda update conda && \
    #conda config --remove channels defaults && \
    conda config --add channels conda-forge

# set up conda environment and add to $PATH
RUN conda create -n cresi python=3.6 \
                    && echo "source activate cresi" > ~/.bashrc
ENV PATH /opt/conda/envs/cresi/bin:$PATH

RUN mkdir -p /root/.torch/models

RUN pip install torch==1.2.0 -f https://download.pytorch.org/whl/torch_stable
RUN pip install torchvision==0.4.0
RUN pip install albumentations==0.4.1
RUN pip install pretrainedmodels

RUN pip install tensorboardX \
	&& pip install torchsummary \
	&& pip install utm \
	&& pip install numba

RUN pip uninstall apex
RUN git clone https://github.com/NVIDIA/apex
RUN sed -i 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' apex/setup.py
RUN  pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"  /apex
RUN conda install -n cresi \
			  gdal=2.4.2 \
		   	  geopandas \
			  fiona \
			  rasterio \
              osmnx=0.10 \
     	      awscli \
              affine \
              pyhamcrest \
              cython \
              fiona \
              h5py \
              ncurses \
              jupyter \
              jupyterlab \
              ipykernel \
              libgdal \
              matplotlib \
	          ncurses \
              numpy \
			  statsmodels \
              #opencv \
              pandas \
              pillow \
              pip \
              scipy \
              scikit-image \
              scikit-learn \
              shapely \
              #gdal \
              rtree \
              testpath \
              tqdm \
              pandas \
              #geopandas \
              #rasterio \
			  #opencv=4.0.0 \
			  opencv \
	&& conda clean -p \
	&& conda clean -t \
	&& conda clean --yes --all
RUN pip install networkx==2.3
RUN pip install pygeoif
ENV LD_LIBRARY_PATH /miniconda/lib:${LD_LIBRARY_PATH}
RUN apt update
RUN apt install openjdk-11-jre-headless -y

# add a jupyter kernel for the conda environment in case it's wanted
RUN source activate cresi && python -m ipykernel.kernelspec
RUN wget http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-b040e4a9b.pth -P /root/.cache/torch/checkpoints/
RUN wget http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth -P /root/.cache/torch/checkpoints/

# open ports for jupyterlab and tensorboard
EXPOSE 8888 6006

WORKDIR /work

COPY . /work/
# get trained model weights
RUN wget -O weights/spacenet_dpn_unet_mc_dpn92_mc_2_best_apls https://www.dropbox.com/s/1fb8l07b2vsrl8b/spacenet_dpn_unet_mc_dpn92_mc_2_best_apls?dl=0
RUN wget -O weights/spacenet_dpn_unet_mc_dpn92_mc_2_best_dice https://www.dropbox.com/s/npi3gz7l3bmcqb6/spacenet_dpn_unet_mc_dpn92_mc_2_best_dice?dl=0
RUN wget -O weights/spacenet_dpn_unet_mc_dpn92_mc_2_last  https://www.dropbox.com/s/jtjl87cxbl03508/spacenet_dpn_unet_mc_dpn92_mc_2_last?dl=0

RUN wget -O weights/spacenet_dpn_unet_dpn92_3_best_apls https://www.dropbox.com/s/xjz15g6456psr2e/spacenet_dpn_unet_dpn92_3_best_apls?dl=0
RUN wget -O weights/spacenet_dpn_unet_dpn92_3_best_dice https://www.dropbox.com/s/iit5em86am5ttpp/spacenet_dpn_unet_dpn92_3_best_dice?dl=0
RUN wget -O weights/spacenet_dpn_unet_dpn92_3_last https://www.dropbox.com/s/i1b019f98okbmfz/spacenet_dpn_unet_dpn92_3_last?dl=0

RUN wget -O weights/spacenet_irv_unet_inceptionresnetv2_0_best_apls https://www.dropbox.com/s/99kepp3w7vrzzor/spacenet_irv_unet_inceptionresnetv2_0_best_apls?dl=0
RUN wget -O weights/spacenet_irv_unet_inceptionresnetv2_0_best_dice https://www.dropbox.com/s/uwv87bcezlx26mo/spacenet_irv_unet_inceptionresnetv2_0_best_dice?dl=0
RUN wget -O weights/spacenet_irv_unet_inceptionresnetv2_0_last https://www.dropbox.com/s/6xl9j5jr8g77pnx/spacenet_irv_unet_inceptionresnetv2_0_last?dl=0

RUN wget -O weights/spacenet_irv_unet_inceptionresnetv2_1_best_apls https://www.dropbox.com/s/ojhbjthben2d9e4/spacenet_irv_unet_inceptionresnetv2_1_best_apls?dl=0
RUN wget -O weights/spacenet_irv_unet_inceptionresnetv2_1_best_dice https://www.dropbox.com/s/srfijcchq6zsjpy/spacenet_irv_unet_inceptionresnetv2_1_best_dice?dl=0
RUN wget -O weights/spacenet_irv_unet_inceptionresnetv2_1_last https://www.dropbox.com/s/p11lry7pnxeml38/spacenet_irv_unet_inceptionresnetv2_1_last?dl=0



RUN chmod 777 train.sh
RUN chmod 777 test.sh

RUN ["/bin/bash"]