FROM smly/pytorch:latest
LABEL maintainer="smly <i@ho.lc>"

# ************************************************************
# For testing, /root/working should be deployed on /wdata/
RUN cd /root &&\
    wget --quiet https://www.dropbox.com/s/3jx8ebabkkw1zb9/r50a_best.zip?dl=0 -O r50a_best.zip &&\
    unzip r50a_best.zip &&\
    wget --quiet https://www.dropbox.com/s/sh0xm7brfh0lxmn/serx50_best.zip?dl=0 -O serx50_best.zip &&\
    unzip serx50_best.zip &&\
    wget --quiet https://www.dropbox.com/s/7wda1oyliglqufc/roads_train_speed_conversion_binned.csv?dl=0 -O roads_train_speed_conversion_binned.csv

WORKDIR /root

# Add codes
ADD *.sh /root/
ADD aa /root/aa
ADD configs /root/configs

RUN chmod a+x test.sh train.sh
