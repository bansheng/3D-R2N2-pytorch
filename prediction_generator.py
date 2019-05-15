# coding=UTF-8
'''
@Description: 用于创建Pytorch版本的网络的逐步重建3D信息
@Author: dingyadong
@Github: https://github.com/bansheng
@LastAuthor: dingyadong
@since: 2019-04-13 09:50:42
@lastTime: 2019-05-13 10:11:28
'''
import os

from demo import main, set_pred_file_name, set_model_name, set_weights
from lib.config import cfg_from_list

Checkpoint_dir = [
    # './output/ResidualGRUNet/default_model',
    # './output/GRUNet/default_model',
    # './output/ResidualGRUNoBNNet/default_model',
    './output/ResidualGRUNet_No_Regularition/default_model'
]
Pre_dir = [
    # './prediction/ResidualGRUNet/',
    # './prediction/GRUNet/',
    # './prediction/ResidualGRUNoBNNet/',
    './prediction/ResidualGRUNet_No_Regularition/',
]

Model_names = [
    # 'ResidualGRUNet',
    # 'GRUNet',
    # 'ResidualGRUNoBNNet',
    'ResidualGRUNet_No_Regularition',
]

if __name__ == "__main__":
    # solver_init()
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    for checkpoint_dir, pre_dir, model_name in zip(Checkpoint_dir, Pre_dir, Model_names):
        # print(checkpoint_dir, pre_dir)
        weights = os.path.join(checkpoint_dir, 'checkpoint.tar')
        set_weights(weights)
        set_model_name(model_name)
        
        # if pre_dir == './prediction/ResidualGRUNet_No_Regularition/':
        #     cfg_from_list(['TEST.VOXEL_THRESH', [0.4]])

        for index in range(1, 11):
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.%d.tar' % (index*2000))
            # print(save_path)
            symlink_path = os.path.join(checkpoint_dir, 'checkpoint.tar')
            if os.path.lexists(symlink_path):
                os.remove(symlink_path)
            os.symlink("%s" % os.path.abspath(checkpoint_path), symlink_path)

            # set_pred_file_name
            name = os.path.join(pre_dir, 'prediction.%d.obj' % (index*2000))

            set_pred_file_name(name)

            main()
            # Make a symlink to the latest network params
            # os.symlink("%s" % os.path.abspath(checkpoint_path), symlink_path)
