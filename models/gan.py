# coding=UTF-8
'''
@Description:
@Author: dingyadong
@Github: https://github.com/bansheng
@LastAuthor: dingyadong
@since: 2019-04-22 16:32:22
@lastTime: 2019-04-22 20:54:45
'''
import datetime as dt
import numpy as np
# import torch
import torch.nn as nn

from lib.utils import weight_init

# Generator1 = nn.Sequential(                      # Discriminator
#     nn.Linear(2, 128),   # receive art work either from the famous artist or a newbie like G with label
#     nn.BatchNorm3d(128),
#     nn.ReLU(),
#     nn.Linear(128, 1),   # receive art work either from the famous artist or a newbie like G with label
#     nn.BatchNorm3d(1),
#     nn.Sigmoid()
# )


class Discriminator(nn.Module):

    def __init__(self, random_seed=dt.datetime.now().microsecond):
        print("\ninitializing \"Discriminator\"")
        super(Discriminator, self).__init__()
        self.rng = np.random.RandomState(random_seed)

        self.l1 = nn.Conv3d(2, 128, 3, stride=1, padding=1) # 2*32*32*32 -> 128*32*32*32
        self.b1 = nn.BatchNorm3d(128)
        self.r1 = nn.ReLU()
        self.l2 = nn.Conv3d(128, 1, 3, stride=1, padding=1) # 128*32*32*32 -> 1*32*32*32
        self.b2 = nn.BatchNorm3d(1)
        self.s1 = nn.Sigmoid()

        self.parameter_init()

    def parameter_init(self):
        #initialize all the parameters of the gru net
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                # """
                # For Conv2d, the shape of the weight is 
                # (out_channels, in_channels, kernel_size[0], kernel_size[1]).
                # For Conv3d, the shape of the weight is 
                # (out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]).
                # """
                w_shape = (m.out_channels, m.in_channels, *m.kernel_size)
                m.weight.data = weight_init(w_shape)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)
            elif isinstance(m, nn.Linear):
                # """
                # For Linear module, the shape of the weight is (out_features, in_features)
                # """
                w_shape = (m.out_features, m.in_features)
                m.weight.data = weight_init(w_shape)
                if m.bias is not None:
                    m.bias.data.fill_(0.1)

    def forward(self, x):
        out = self.l1(x)
        out = self.b1(out)
        out = self.r1(out)
        out = self.l2(out)
        out = self.b2(out)
        out = self.s1(out)
        return out