#!/bin/bash

CaffeRoot=~/Codes/Caffe
model=models/yumi_pick_vae

rHome=/mnt/md0/algh/
rCaffeRoot=$rHome/caffe
remote=$beorn
# ...

#### update all scripts
scp -r $CaffeRoot/$model/*.sh $remote:$rCaffeRoot/$model/
scp -r $CaffeRoot/$model/*.prototxt $remote:$rCaffeRoot/$model/
#scp -r $CaffeRoot/$model/*.txt $remote:$rCaffeRoot/$model/

#### download trained caffemodel
#scp $remote:$rCaffeRoot/$model/snapshots/*_vae_iter_1000000.caffemodel $CaffeRoot/$model/snapshots

#### upload trained models
#scp $CaffeRoot/$model/snapshots/10d_vae_iter_1200000.caffemodel $remote:$rCaffeRoot/$model/snapshots/

#### upload training data
scp -r  $CaffeRoot/$model/data/trajs.h5 $remote:$rCaffeRoot/$model/data/
scp -r  $CaffeRoot/$model/data/traindata.txt $remote:$rCaffeRoot/$model/data/

# ...
