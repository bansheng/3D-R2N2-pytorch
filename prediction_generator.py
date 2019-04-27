import os

from demo import main, set_pred_file_name
from lib.config import cfg_from_list

Checkpoint_dir = [
                #   './output/ResidualGRUNet/default_model',
                  './output/GRUNet/default_model',
                  './output/ResidualGRUNoBNNet/default_model',
                #   './output/ResidualGRUNet_theano/default_model',
                  './output/ResidualGRUNet_No_Regularition/default_model'
                  ]
Pre_dir = [
        #    './prediction/ResidualGRUNet/',
           './prediction/GRUNet/',
           './prediction/ResidualGRUNoBNNet/',
        #    './prediction/ResidualGRUNet_theano/',
           './prediction/ResidualGRUNet_No_Regularition/',
           ]

if __name__ == "__main__":
    # solver_init()
    for checkpoint_dir, pre_dir in zip(Checkpoint_dir, Pre_dir):
        # print(checkpoint_dir, pre_dir)
        weights = os.path.join(checkpoint_dir, 'checkpoint.tar')
        # print(weights)
        for index in range(1, 11):
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.%d.tar' % (index*2000))
            # print(save_path)
            symlink_path = os.path.join(checkpoint_dir, 'checkpoint.tar')
            if os.path.lexists(symlink_path):
                os.remove(symlink_path)
            os.symlink("%s" % os.path.abspath(checkpoint_path), symlink_path)

            # set batch_size = 1
            cfg_from_list(['CONST.BATCH_SIZE', 1])

            # set_pred_file_name
            name = os.path.join(pre_dir, 'prediction.%d.obj' % (index*2000))

            set_pred_file_name(name, weights)

            main()
            # Make a symlink to the latest network params
            # os.symlink("%s" % os.path.abspath(checkpoint_path), symlink_path)
