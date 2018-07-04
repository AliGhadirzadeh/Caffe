#!/bin/bash

# ...
dim=5d
model=models/yumi_pick_vae
iter=_vae_iter_750000
i=0
while [ $i -lt "1000" ]
do
  printf -v j "%04d" $i
  python $model/scripts/traj_utils.py scale_traj $model/data/trajs/$j.txt
  ./build/examples/cpp_classification/caffe_forward.bin $model/train_$dim.prototxt $model/snapshots/$dim$iter.caffemodel NO_OPERATION NULL $model/results/traj.txt $model/results
  cp $model/results/o_00_ch_00.txt $model/results/latent_var/$j.txt
  cp $model/results/o_01_ch_00.txt $model/results/traj_restored/$j.txt
  let "i++"
done

# ...
