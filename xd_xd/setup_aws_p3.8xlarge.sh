# AMI ID
# ubuntu/images/hvm-ssd/ubuntu-bionic-18.04-amd64-server-20191002 (ami-06d51e91cea0dac8d)

sudo apt -y update
sudo apt -y upgrade
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt -y update
sudo apt -y install cuda cuda-drivers

sudo apt -y update
sudo apt -y upgrade  # cuda 10.1

sudo apt -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt -y update
sudo apt -y install docker-ce docker-ce-cli containerd.io

sudo reboot

# ------------------------------------------------------------------------------

echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
exec bash

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt -y update && sudo apt -y install nvidia-container-toolkit
sudo systemctl restart docker

sudo usermod -aG docker ubuntu
newgrp docker

# docker --version
# > Docker version 19.03.2, build 6a30dfc

# docker run --gpus all nvidia/cuda:10.0-base nvidia-smi

# cd docker/pytorch
# docker build -t smly/pytorch .
# docker push smly/pytorch:latest
# docker run --gpus all -it --rm smly/pytorch bash
