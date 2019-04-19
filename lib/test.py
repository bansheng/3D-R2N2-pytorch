from binvox_rw import read_as_3d_array
import torch
import numpy as np

with open('/Users/dingyadong/Documents/GitHub/01-3D-R2N2/ShapeNet/ShapeNetVox32/02828884/1a55e418c61e9dab6f4a86fe50d4c8f0/model.binvox', 'rb') as f:
    voxel = read_as_3d_array(f)
    # print(voxel.data)
    voxel_data = voxel.data
    batch_voxel1 = voxel_data < 1
    batch_voxel2 = voxel_data
    print(batch_voxel2 == 0)

