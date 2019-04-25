import os

from demo import main, set_pred_file_name
from lib.config import cfg, cfg_from_list

checkpoint_dir = './output/ResidualGRUNet/default_model'
pre_dir = './prediction/ResidualGRUNet/'

if __name__ == "__main__":
    # solver_init()
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

        set_pred_file_name(name)

        main()
        # Make a symlink to the latest network params
        # os.symlink("%s" % os.path.abspath(checkpoint_path), symlink_path)
