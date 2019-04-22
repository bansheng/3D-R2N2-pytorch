# coding=UTF-8
'''
@Description:
@Author: dingyadong
@Github: https://github.com/bansheng
@LastAuthor: dingyadong
@since: 2019-04-22 16:32:22
@lastTime: 2019-04-22 16:34:58
'''

import numpy as np
# import torch
import torch.nn as nn
# from torch.autograd import Variable
from torch.nn import (BatchNorm2d, BatchNorm3d, Conv2d, Conv3d, LeakyReLU,
                      Linear, MaxPool2d, Sigmoid, Tanh)

D = nn.Sequential(                      # Discriminator
    nn.Linear(ART_COMPONENTS+1, 128),   # receive art work either from the famous artist or a newbie like G with label
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),                       # tell the probability that the art work is made by artist
)

opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()   # something about continuous plotting

for step in range(10000):
    artist_paintings, labels = artist_works_with_labels()           # real painting, label from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)                      # random ideas
    G_inputs = torch.cat((G_ideas, labels), 1)                      # ideas with labels
    G_paintings = G(G_inputs)                                       # fake painting w.r.t label from G

    D_inputs0 = torch.cat((artist_paintings, labels), 1)            # all have their labels
    D_inputs1 = torch.cat((G_paintings, labels), 1)
    prob_artist0 = D(D_inputs0)                 # D try to increase this prob
    prob_artist1 = D(D_inputs1)                 # D try to reduce this prob

    D_score0 = torch.log(prob_artist0)          # maximise this for D
    D_score1 = torch.log(1. - prob_artist1)     # maximise this for D
    D_loss = - torch.mean(D_score0 + D_score1)  # minimise the negative of both two above for D
    G_loss = torch.mean(D_score1)               # minimise D score w.r.t G

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)      # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()