#!/bin/bash

# ...
model=models/yumi_pick_vae
dim=5d
iter=_vae_iter_1200000
i=0
# functions start here
printf -v j "%04d" $i
python $model/scripts/traj_utils.py scale_traj $model/data/trajs/$j.txt
./build/examples/cpp_classification/caffe_forward.bin \
$model/train_$dim.prototxt \
$model/snapshots/$dim$iter.caffemodel \
NO_OPERATION \
NULL \
$model/results/traj.txt \
$model/results
python $model/scripts/traj_utils.py evaluate $model/results
# ...
