import os

import matplotlib.pyplot as plt
import numpy as np

Save_dirs = [
    './output/ResidualGRUNet/default_model',
    './output/GRUNet/default_model',
    # './output/ResidualGRUNoBNNet/default_model',
    # './output/ResidualGRUNet_theano/default_model',
    # './output/ResidualGRUNet_No_Regularition/default_model'
]

Labels = [
    'ResidualGRUNet',
    'GRUNet',
    # 'ResidualGRUNoBNNet',
    # 'ResidualGRUNet_theano',
    # 'ResidualGRUNet_No_Regularition'
]

font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}

def file_read(files):
    # files [] 文件名列表
    ylines = []
    for save_dir in Save_dirs:
        for filename in files:
            ylist = []
            with open(os.path.join(save_dir, filename), 'r') as f:
                ylist = f.readlines()
            for y in ylist:
                y = y.strip()
            ylines.append(ylist[:201:1])
    return ylines

if __name__ == '__main__':

    plt.figure(1, figsize=(12, 5))
    lines = file_read(['loss.20000.txt'])
    colors = ['red', 'blue', 'green', 'purple', 'black']
    # lines[0] = lines[0][:2001]
    xmax = 0
    for id, axis_y in enumerate(lines):
        num_of_points = len(axis_y)
        xmax = max(xmax, num_of_points-1)
        axis_x = np.linspace(0, num_of_points-1, num_of_points, dtype=np.float32) #0-10000
        axis_y = np.array(axis_y, dtype=np.float32)
        # print(axis_x.shape, axis_y.shape)
        plt.plot(axis_x, axis_y.flatten(), color=colors[id], linewidth=1, label=Labels[id])
        print(colors[id])

    plt.legend(loc='upper right', prop=font)
    plt.xlim((0, xmax))
    plt.ylim((0, 1))
    plt.xlabel('train iter', font)
    plt.ylabel('loss', font)
    new_ticks = np.linspace(0, xmax, 5)
    plt.xticks(new_ticks)
    plt.yticks([0, 0.5, 1])
    plt.tick_params(labelsize=14)
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.show()
