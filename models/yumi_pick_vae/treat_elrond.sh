#!/bin/bash

CaffeRoot=~/Codes/Caffe
model=models/yumi_pick_vae

rHome=/mnt/md0/ali/
rCaffeRoot=$rHome/codes/KTH/caffe
remote=$elrond
# ...

#### update all scripts
scp -r $CaffeRoot/$model/*.sh $remote:$rCaffeRoot/$model/
scp -r $CaffeRoot/$model/*.prototxt $remote:$rCaffeRoot/$model/
#scp -r $CaffeRoot/$model/*.txt $remote:$rCaffeRoot/$model/

#### download trained caffemodel
#scp $remote:$rCaffeRoot/$model/snapshots/*_vae_iter_750000.caffemodel $CaffeRoot/$model/snapshots
#scp $remote:$rCaffeRoot/$model/snapshots/*_vae_iter_750000.solverstate $CaffeRoot/$model/snapshots

#### upload trained models
#scp $CaffeRoot/$model/snapshots/5d_vae_iter_750000.caffemodel $remote:$rCaffeRoot/$model/snapshots/

#### upload training data
#scp -r  $CaffeRoot/$model/data/trajs.h5 $remote:$rCaffeRoot/$model/data/
#scp -r  $CaffeRoot/$model/data/traindata.txt $remote:$rCaffeRoot/$model/data/

# ...
