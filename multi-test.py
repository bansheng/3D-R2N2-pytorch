# coding=UTF-8
'''
@Description: 用于多视图多物体重建
@Author: dingyadong
@Github: https://github.com/bansheng
@LastAuthor: dingyadong
@since: 2019-05-13 09:44:28
@lastTime: 2019-05-13 11:09:02
'''
import os
import shutil

from demo import main, set_pred_file_name, load_demo_images
from lib.config import cfg_from_list


def make_path(p):
    if os.path.exists(p):       # 判断文件夹是否存在
        shutil.rmtree(p)        # 删除文件夹
    os.mkdir(p)                 # 创建文件夹

rootPath = './multitest/'


if __name__ == "__main__":
    cfg_from_list(['CONST.BATCH_SIZE', 1])
    make_path(rootPath)
    for i in range(11):
        imgs = './imgs/test%d/0' % (i)
        pre_dir = rootPath+'test%d' % (i)
        make_path(pre_dir)
        
        for j in range(5):
            load_demo_images(imgs, maxrange=j+1)
            name = os.path.join(pre_dir, 'prediction%d.obj' % (j))
            set_pred_file_name(name)
            main()
