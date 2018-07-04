import h5py
import os
import numpy as np
import math
import os.path
import random
import matplotlib.pyplot as plt

root_dir = "/home/ali/Codes/Caffe/models/yumi_pick_vae/results/"
data_type = "traj_restored/"
n_data = 1
traj_length = 77
n_joints = 7

trajs = np.zeros((n_data, n_joints*traj_length))

for i in range(0, n_data):
    data_filename = root_dir + data_type + '{:0>4}'.format(i) + '.txt'
    freader = open(data_filename, 'r')
    freader.readline()
    freader.readline()       # number of joints
    for j in range(0, n_joints):
        for t in range(0, traj_length ):
            trajs[i,t*n_joints+j] = float(freader.readline())

print trajs.shape
print np.amax(trajs)
print np.amin(trajs)

for d in range(0,n_data):
    x = np.zeros((traj_length))
    y = np.zeros((n_joints, traj_length))
    for j in range(0, n_joints):
        for t in range(0, traj_length):
            x[t] = t
            y[j,t] = trajs[d,t*n_joints+j]
        plt.plot(x, y[j], '-')
plt.show()
