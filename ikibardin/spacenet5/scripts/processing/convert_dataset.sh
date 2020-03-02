#!/usr/bin/env bash

set -e

# Spacenet 3 cities

VEGAS_IMAGES="/data/train/AOI_2_Vegas/PS-RGB"
VEGAS_ROADS="/data/train/AOI_2_Vegas/SN3_roads_train_AOI_2_Vegas_road_labels.csv"

PARIS_IMAGES="/data/train/AOI_3_Paris/PS-RGB"
PARIS_ROADS="/data/train/AOI_3_Paris/SN3_roads_train_AOI_3_Paris_road_labels.csv"

SHANGHAI_IMAGES="/data/train/AOI_4_Shanghai/PS-RGB"
SHANGHAI_ROADS="/data/train/AOI_4_Shanghai/SN3_roads_train_AOI_4_Shanghai_road_labels.csv"

KHARTOUM_IMAGES="/data/train/AOI_5_Khartoum/PS-RGB"
KHARTOUM_ROADS="/data/train/AOI_5_Khartoum/SN3_roads_train_AOI_5_Khartoum_road_labels.csv"

# Spacenet 5 cities

MOSCOW_IMAGES="/data/train/AOI_7_Moscow/PS-RGB"
MOSCOW_ROADS="/data/train/AOI_7_Moscow/train_AOI_7_Moscow_geojson_roads_speed_wkt_weighted_raw.csv"

MUMBAI_IMAGES="/data/train/AOI_8_Mumbai/PS-RGB"
MUMBAI_ROADS="/data/train/AOI_8_Mumbai/train_AOI_8_Mumbai_geojson_roads_speed_wkt_weighted_raw.csv"

OUTPUT="/wdata/converted"


python -m src.processing.convert_dataset \
    --images ${VEGAS_IMAGES} \
    --roads ${VEGAS_ROADS} \
    --output ${OUTPUT};

#python -m src.processing.convert_dataset \
#    --images ${PARIS_IMAGES} \
#    --roads ${PARIS_ROADS} \
#    --output ${OUTPUT};
#
#python -m src.processing.convert_dataset \
#    --images ${SHANGHAI_IMAGES} \
#    --roads ${SHANGHAI_ROADS} \
#    --output ${OUTPUT};
#
#python -m src.processing.convert_dataset \
#    --images ${KHARTOUM_IMAGES} \
#    --roads ${KHARTOUM_ROADS} \
#    --output ${OUTPUT};


#
#python -m src.processing.convert_dataset \
#    --images ${MOSCOW_IMAGES} \
#    --roads ${MOSCOW_ROADS} \
#    --output ${OUTPUT};
#
#
#python -m src.processing.convert_dataset \
#    --images ${MUMBAI_IMAGES} \
#    --roads ${MUMBAI_ROADS} \
#    --output ${OUTPUT};
