FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# Use a fixed apt-get repo to stop intermittent failures due to flaky httpredir connections,
# as described by Lionel Chan at http://stackoverflow.com/a/37426929/5881346
RUN sed -i "s/httpredir.debian.org/debian.uchicago.edu/" /etc/apt/sources.list && \
    apt-get update && apt-get install -y build-essential
    
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion zip unzip
    
# install Anaconda3
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.2.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y libglu1

RUN conda install -c conda-forge opencv \
 	tqdm \
	shapely \
	# pytorch torchvision cudatoolkit=10.1 -c pytorch \
	numpy \
	gdal=2.4.2 

RUN pip install imgaug

RUN pip install lightgbm

RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

RUN pip install utm
RUN pip install pandas

RUN conda install -c conda-forge pyproj

# RUN conda update numpy

# RUN conda install -c conda-forge gdal=2.4.2 


WORKDIR /work

# copy entire directory where docker file is into docker container at /work
COPY . /work/

RUN chmod 777 train.sh
RUN chmod 777 test.sh
RUN chmod 777 download_models.sh

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/work/download_models.sh" ]