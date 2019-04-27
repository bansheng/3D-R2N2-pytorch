# coding=UTF-8
'''
@Description: 
@Author: dingyadong
@Github: https://github.com/bansheng
@LastAuthor: dingyadong
@since: 2019-04-17 11:23:11
@lastTime: 2019-04-27 15:05:43
'''
import os
import shutil
import sys
from subprocess import call

# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from lib.config import cfg, cfg_from_list
from lib.data_augmentation import preprocess_img
from lib.solver import Solver
from lib.voxel import voxel2obj
from models import load_model

# import torch

'''
Demo code for the paper

Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object
Reconstruction, ECCV 2016
'''

if sys.version_info < (3, 0):
    raise Exception(
        "Please follow the installation \
            instruction on 'https://github.com/bansheng/3D-R2N2-pytorch'")


DEFAULT_WEIGHTS = 'output/ResidualGRUNet/default_model/checkpoint.tar'

pred_file_name = ''

def set_pred_file_name(name, weights):
    global pred_file_name, DEFAULT_WEIGHTS
    pred_file_name = name
    DEFAULT_WEIGHTS = weights


def cmd_exists(cmd):
    return shutil.which(cmd) is not None


def download_model(fn):
    if not os.path.isfile(fn):
        # Download the file if doewn't exist
        print('Downloading a pretrained model')
        call(['curl', 'ftp://cs.stanford.edu/cs/cvgl/ResidualGRUNet.npy',
              '--create-dirs', '-o', fn])
    else:
        print('A pretrained model detected!')


def load_demo_images():
    ims = []
    for i in range(3):
        im = preprocess_img(Image.open('imgs/%d.jpg' % #进来的时候是127*127*3
                                       i).resize((127, 127)), train=False)
        ims.append([np.array(im).transpose(
            (2, 0, 1)).astype(np.float32)])
        # plt.imshow(im)
        # plt.show()
    return np.array(ims)


def main():
    '''Main demo function'''
    # Save prediction into a file named 'prediction.obj' or the given argument
    global pred_file_name
    if not cfg.TEST.MULTITEST or pred_file_name == '':
        pred_file_name = sys.argv[1] if len(sys.argv) > 1 else 'prediction.obj'

    demo_imgs = load_demo_images()

    # Download and load pretrained weights
    download_model(DEFAULT_WEIGHTS)

    # Use the default network model
    NetClass = load_model('ResidualGRUNet')
    
    # print(NetClass)

    # Define a network and a solver. Solver provides a wrapper for the test function.
    net = NetClass()  # instantiate a network
    solver = Solver(net)                # instantiate a solver
    
    solver.load(DEFAULT_WEIGHTS)        # load pretrained weights
    # solver.graph_view(demo_imgs)
    # return
    # Run the network
    voxel_prediction, _ = solver.test_output(demo_imgs)
    # Save the prediction to an OBJ file (mesh file).
    # (self.batch_size, 2, n_vox, n_vox, n_vox)
    # print((voxel_prediction[0, 1, :, :, :] > cfg.TEST.VOXEL_THRESH))
    # print(type(voxel_prediction[0, :, 1, :, :].data.numpy()))
    # print(type(cfg.TEST.VOXEL_THRESH))
    voxel2obj(pred_file_name,
              voxel_prediction[0, 1, :, :, :].data.numpy() > cfg.TEST.VOXEL_THRESH) # modified

    # Use meshlab or other mesh viewers to visualize the prediction.
    # For Ubuntu>=14.04, you can install meshlab using
    # `sudo apt-get install meshlab`
    print('writing voxel to %s' % (pred_file_name))
    if cfg.TEST.CALL_MESHLAB: #需要打开meshlab
        if cmd_exists('meshlab'):
            call(['meshlab', pred_file_name])
        else:
            print('Meshlab not found: please use visualization of your choice to view %s' %
                    pred_file_name)


if __name__ == '__main__':
    # Set the batch size to 1
    cfg_from_list(['CONST.BATCH_SIZE', 1])

    # solver_init()
    main()
