import os

import matplotlib.pyplot as plt
import numpy as np

save_dir = 'output/ResidualGRUNet/default_model/'


def file_read(files):
    # files [] 文件名列表
    ylines = []
    for filename in files:
        ylist = []
        with open(os.path.join(save_dir, filename), 'r') as f:
            ylist = f.readlines()
        for y in ylist:
            y = y.strip()
        ylines.append(ylist)
    return ylines

if __name__ == '__main__':

    plt.figure(1, figsize=(12, 10))
    lines = file_read(['loss.10000.txt'])
    colors = ['red', 'blue', 'green', 'yellow']
    # lines[0] = lines[0][:2001]
    xmax = 0
    for id, axis_y in enumerate(lines):
        num_of_points = len(axis_y)
        xmax = max(xmax, num_of_points-1)
        axis_x = np.linspace(0, num_of_points-1, num_of_points, dtype=np.float32) #0-10000
        axis_y = np.array(axis_y, dtype=np.float32)
        # print(axis_x.shape, axis_y.shape)
        plt.plot(axis_x, axis_y.flatten(), color=colors[id], linewidth=0.5)
        print(colors[id])

    plt.xlim((0, xmax))
    plt.ylim((0, 1))
    plt.xlabel('train iter')
    plt.ylabel('loss')
    new_ticks = np.linspace(0, xmax, 5)
    plt.xticks(new_ticks)
    plt.yticks([0, 0.5, 1])
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    plt.show()
