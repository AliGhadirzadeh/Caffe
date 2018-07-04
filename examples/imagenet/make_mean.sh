#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=data/table_objects
DATA=data/table_objects
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/PikachuRandTr_200_lmdb \
  $DATA/PikachuRandTr_200.binaryproto

echo "Done."
