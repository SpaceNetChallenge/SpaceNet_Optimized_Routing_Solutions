#!/usr/bin/env bash

set -e

# Spacenet 5 cities

MUMBAI_IMAGES="/wdata/test/AOI_8_Mumbai/PS-RGB"

SANJUAN_IMAGES="/wdata/test/AOI_9_San_Juan/PS-RGB"

MOSCOW_IMAGES="/wdata/test/AOI_7_Moscow/PS-RGB"

OUTPUT="/wdata/test/converted"


python -m src.processing.convert_dataset \
    --images ${MUMBAI_IMAGES} \
    --output ${OUTPUT};

python -m src.processing.convert_dataset \
    --images ${SANJUAN_IMAGES} \
    --output ${OUTPUT};

python -m src.processing.convert_dataset \
    --images ${MOSCOW_IMAGES} \
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
