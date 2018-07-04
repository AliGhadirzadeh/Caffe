import numpy as np
from scipy import interpolate
import h5py
import random
import sys


class Traj:
    def __init__(self):
        self.dir2params = "/home/ali/Codes/Caffe/models/yumi_pick_vae/scripts/traj_params.txt"
        self.ntrajs = 0
        self.njoints = 7

        # samples
        self.samples = []               #sampled joint positions
        self.samples_time = []          #time instances of sampled points
        self.nsamples = 0               #number of sampled points
        self.sampling_period = 0.1      # in seconds
        self.sample_buffer = []         # a buffer to store loaded samples
        self.samples_max = []
        self.samples_min = []

        # trajectories
        self.trajs_duration_max = 7.651 # in seconds
        self.trajs_ndata = []
        self.trajs_time = []            #time instances of the trajectory
        self.trajs_pos = []             # joint position values
        self.trajs_vel = []             # joint velocity values
        self.trajs_acc = []             # joint acceleration values

    def read_traj(self, filename):
        #TODO: change this function to read from XML file
        freader = open(filename, 'r')
        freader.readline()
        ndata = int(freader.readline())
        freader.readline()
        t = []
        for i in range(ndata):
            t.append(float(freader.readline()))
        pos = [[0 for ii in range(ndata)] for jj in range(self.njoints)]
        vel = [[0 for ii in range(ndata)] for jj in range(self.njoints)]
        acc = [[0 for ii in range(ndata)] for jj in range(self.njoints)]
        for j in range(7): # number of joints
            freader.readline() #<JOINT NAME>
            freader.readline() #joint name
            freader.readline() #<POSITION>
            for i in range(ndata):
                pos[j][i] = float(freader.readline())
            freader.readline() #<VELOCITY>
            for i in range(ndata):
                vel[j][i] = float(freader.readline())
            freader.readline() #<ACCELERATION>
            for i in range(ndata):
                acc[j][i] = float(freader.readline())
        # adding the fixed points
        while (t[ndata-1]+self.sampling_period) < self.trajs_duration_max:
            t.append(t[ndata-1]+self.sampling_period)
            for j in range(self.njoints):
                pos[j].append(pos[j][ndata-1])
                vel[j].append(pos[j][ndata-1])
                acc[j].append(pos[j][ndata-1])
            ndata = ndata + 1
        self.trajs_pos.append(pos)
        self.trajs_vel.append(vel)
        self.trajs_acc.append(acc)
        self.trajs_time.append(t)
        self.trajs_ndata.append(ndata)
        self.ntrajs += 1
    def save_sample(self, filename, sample_idx):
        fwriter = open(filename, 'w')
        fwriter.write(str(self.nsamples*self.njoints)+"\n")
        fwriter.write(str(1)+"\n")
        for j in range(self.njoints):
            for t in range(self.nsamples):
                fwriter.write(str(self.samples[sample_idx][j][t])+"\n")
        fwriter.close()
    def load_sample(self, filename):
        freader = open(filename, 'r')
        freader.readline()
        freader.readline()
        self.sample_buffer = []
        for j in range(self.njoints):
            s = []
            for t in range(self.nsamples):
                s.extend( [float(freader.readline())] )
            self.sample_buffer.append(s)
        freader.close()

    def plot(self, indices, joints):
        for i in indices:
            for j in joints:
                #plt.plot(self.trajs_time[idx], self.trajs_pos[idx][j], '.')
                if i == -1:
                    plt.plot(self.samples_time, self.sample_buffer[j], 'b-')
                else:
                    plt.plot(self.samples_time, self.samples[i][j], 'g-')
        plt.show()

    def interpolate(self):
        t = np.arange(0, self.trajs_duration_max, 0.05)
        print t
        for j in range(self.njoints):
            plt.plot(self.trajs_time,self.trajs_pos[j], '.' )
            tck = interpolate.splrep(self.trajs_time, self.trajs_pos[j])
            pos = interpolate.splev(t, tck, der=0)
            plt.plot(t, pos, '-')
            #vel = interpolate.splev(t, tck, der=1)
            #plt.plot(t, vel, '-')
        plt.show()
    def sample(self):
        self.samples_time = np.arange(0, self.trajs_duration_max, self.sampling_period)
        self.nsamples = len(self.samples_time)
        self.samples = []
        for i in range(self.ntrajs):
            pos=[]
            for j in range(self.njoints):
                tck = interpolate.splrep(self.trajs_time[i], self.trajs_pos[i][j])
                pos.append(interpolate.splev(self.samples_time, tck, der=0).tolist())
            self.samples.append(pos)
        self.samples_max = [max([self.samples[x][j][y] for x in range(self.ntrajs) for y in range(self.nsamples)]) for j in range(self.njoints)]
        self.samples_min = [min([self.samples[x][j][y] for x in range(self.ntrajs) for y in range(self.nsamples)]) for j in range(self.njoints)]

    def scale(self):
        for traj in range(self.ntrajs):
            for j in range(self.njoints):
                for i in range(self.nsamples):
                    self.samples[traj][j][i] = (self.samples[traj][j][i] - self.samples_min[j])/ \
                                                   (self.samples_max[j] - self.samples_min[j])
    def save_params(self):
        fwriter = open(self.dir2params, 'w')
        for j in range(self.njoints):
            fwriter.write(str(self.samples_max[j])+"\n")
            fwriter.write(str(self.samples_min[j])+"\n")
        fwriter.close()
    def load_params(self):
        freader = open(self.dir2params, 'r')
        self.samples_max = [0 for i in range(self.njoints)]
        self.samples_min = [0 for i in range(self.njoints)]
        for j in range(self.njoints):
            self.samples_max[j] = float(freader.readline())
            self.samples_min[j] = float(freader.readline())
        freader.close()
def main():
    root_dir = "/home/ali/Codes/Caffe/models/yumi_pick_vae/data/trajs/"
    ndata = 10000
    traj = Traj()
    for traj_idx in range(ndata):
        traj.read_traj(root_dir + '{:0>4}'.format(traj_idx) + '.txt')
    traj.sample()
    traj.scale()
    #traj.plot(range(ndata),[0])
    trajs = []
    for d in range(ndata):
        row =[]
        for j in range(7):
            row.extend(traj.samples[d][j])
        trajs.append(row)

    traj.save_params()
    with h5py.File('trajs.h5', 'w') as f:
        f['trajs'] = np.array(trajs)

def scale_traj(filename):
    save_dir = "/home/ali/Codes/Caffe/models/yumi_pick_vae/results/traj.txt"
    traj = Traj()
    traj.read_traj(filename)
    traj.sample()
    traj.load_params()
    traj.scale()
    traj.save_sample(save_dir, 0)

def evaluate(arg):
    root_dir = "/home/ali/Codes/Caffe/models/yumi_pick_vae/data/trajs/"
    result_dir = "/home/ali/Codes/Caffe/models/yumi_pick_vae/results/"
    # Visual check: reconstructed trajectories vs originals
    if True:
        traj = Traj()
        for i in range(100):
            traj.read_traj(root_dir + '{:0>4}'.format(i) + '.txt')
        traj.sample()
        traj.load_params()
        traj.scale()
        for i in [25, 50]:
            traj.load_sample(result_dir+"traj_restored/"+'{:0>4}'.format(i) + '.txt')
            traj.plot([i,-1],range(7))
    # plot a bunch of trajs
    if False:
        ndata = 1
        traj = Traj()
        for traj_idx in range(ndata):
            traj.read_traj(root_dir + '{:0>4}'.format(traj_idx) + '.txt')
        traj.sample()
        traj.load_params()
        traj.scale()
        traj.plot(range(ndata), range(7))
    if True:
        ndata=1000
        traj = Traj()
        for traj_idx in range(ndata):
            traj.read_traj(root_dir + '{:0>4}'.format(traj_idx) + '.txt')
        traj.sample()
        traj.load_params()
        traj.scale()
        err_sum = 0
        for i in range(ndata):
            traj.load_sample(result_dir+"traj_restored/"+'{:0>4}'.format(i) + '.txt')
            for j in range(traj.njoints):
                for t in range(traj.nsamples):
                    err = (traj.samples[i][j][t] - traj.sample_buffer[j][t])
                    err_sum += err*err
        print err_sum/(2*ndata)




if __name__ == "__main__":
    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    if sys.argv[1] != "scale_traj":
        import matplotlib.pyplot as plt
    if len(sys.argv) < 2:
        main()
    else:
        globals()[sys.argv[1]](sys.argv[2])
