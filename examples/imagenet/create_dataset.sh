#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

SET=reach_action
FOLDER=crop
HEIGHT1=0
WIDTH1=0

EXAMPLE=data/$SET
DATA=data/$SET
TOOLS=build/tools

TRAIN_DATA_ROOT=/home/ali/Codes/Caffe/

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating first lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$HEIGHT1 \
    --resize_width=$WIDTH1 \
    --gray \
    $TRAIN_DATA_ROOT \
    $DATA/$FOLDER.txt \
    $EXAMPLE/renameme_lmdb
    #$EXAMPLE/{$FOLDER}{$HEIGHT1}X{$WIDTH1}_lmdb


#echo "Creating second lmdb..."
#GLOG_logtostderr=1 $TOOLS/convert_imageset \
#    --resize_height=$HEIGHT2 \
#    --resize_width=$WIDTH2 \
#    --gray\
#    $TRAIN_DATA_ROOT \
#    $DATA/$FOLDER.txt \
#    $EXAMPLE/{$FOLDER}{$HEIGHT2}X{$WIDTH2}Gray_lmdb

echo "Done."
