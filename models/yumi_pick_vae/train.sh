#!/usr/bin/env sh
set -e

./build/tools/caffe train \
  --solver=models/yumi_pick_vae/solver.prototxt  --weights=models/yumi_pick_vae/snapshots/5d_vae_iter_750000.caffemodel$@


  #-snapshot=models/yumi_pick_vae/snapshots/5d_vae_iter_1200000.solverstate
