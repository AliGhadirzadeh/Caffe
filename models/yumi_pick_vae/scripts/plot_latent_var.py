import h5py
import os
import numpy as np
import math
import os.path
import random
import matplotlib.pyplot as plt


root_dir = "/home/ali/Codes/Caffe/models/yumi_pick_vae/results/"
n_data = 1000
d_latent_var = 5

x_axis_data = np.zeros((n_data))
y_axis_data = np.zeros((n_data))

var = np.zeros((d_latent_var))

for m in range(0, d_latent_var):
    for n in range(0,m):
        for i in range(0, n_data):
            filename = root_dir + "latent_var/" + '{:0>4}'.format(i) + '.txt'
            freader = open(filename, 'r')
            freader.readline()
            freader.readline()
            for j in range(0, d_latent_var):
                var[j] = float(freader.readline())
            freader.close()
            x_axis_data[i] = var[m]
            y_axis_data[i] = var[n]

        plt.plot(x_axis_data, y_axis_data, '.')
        plt.show()
