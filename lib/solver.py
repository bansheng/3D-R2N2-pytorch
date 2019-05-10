# coding=UTF-8
'''
@Description:
@Author: dingyadong
@Github: https://github.com/bansheng
@LastAuthor: dingyadong
@since: 2019-04-17 11:23:11
@lastTime: 2019-05-10 13:12:38
'''
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam, lr_scheduler

from lib.config import cfg
from lib.utils import Timer, has_nan


def max_or_nan(params):
    params = list(params)
    nan_or_max_param = torch.FloatTensor(len(params)).zero_()
    if torch.cuda.is_available():
        nan_or_max_param = nan_or_max_param.type(torch.cuda.FloatTensor)

    for param_idx, param in enumerate(params):
        # If there is nan, max will return nan
        # Note that param is Variable
        nan_or_max_param[param_idx] = torch.max(torch.abs(param)).data
        print('param %d : %f' % (param_idx, nan_or_max_param[param_idx]))
    return nan_or_max_param


class Solver(object):
    def __init__(self, net):
        self.net = net
        self.lr = cfg.TRAIN.DEFAULT_LEARNING_RATE
        print('Set the learning rate to %f.' % self.lr)

        #set the optimizer
        self.set_optimizer(cfg.TRAIN.POLICY)

    def set_optimizer(self, policy=cfg.TRAIN.POLICY):
        """
        This function is used to set the optimization algorithm of training 
        """
        net = self.net
        lr = self.lr
        if cfg.TRAIN.NORMALIZATION:
            w_decay = cfg.TRAIN.WEIGHT_DECAY
        else:
            w_decay = 0.
        if policy == 'sgd':
            momentum = cfg.TRAIN.MOMENTUM
            self.optimizer = SGD(
                net.parameters(),
                lr=lr,
                weight_decay=w_decay,
                momentum=momentum)
        elif policy == 'adam':
            self.optimizer = Adam(
                net.parameters(), lr=lr, weight_decay=w_decay)
        else:
            sys.exit('Error: Unimplemented optimization policy')

    def train_loss(self, x, y):
        """
        y is provided and test is False, only the loss will be returned.
        """
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        if torch.cuda.is_available():
            x.cuda(async=True)
            y.cuda(async=True)
            x = x.type(torch.cuda.FloatTensor)
            y = y.type(torch.cuda.FloatTensor)

        x = Variable(x, requires_grad=False)
        y = Variable(y, requires_grad=False)

        loss = self.net(x, y, test=False)

        #compute gradient and do parameter update step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_queue, val_queue=None):
        ''' Given data queues, train the network '''
        # Parameter directory
        save_dir = os.path.join(cfg.DIR.OUT_PATH)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Timer for the training op and parallel data loading op.
        train_timer = Timer()
        data_timer = Timer()
        training_losses = []

        # Setup learning rates
        lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]

        #Setup the lr_scheduler
        self.lr_scheduler = lr_scheduler.MultiStepLR(
            self.optimizer, lr_steps, gamma=0.1)  # gamma为下降系数

        start_iter = 0
        # Resume training
        if cfg.TRAIN.RESUME_TRAIN:
            self.load(cfg.CONST.WEIGHTS)
            start_iter = cfg.TRAIN.INITIAL_ITERATION

        if cfg.TRAIN.SHOW_LOSS:  # 要动态打印
            import matplotlib.pyplot as plot
            plot.figure(1, figsize=(12, 5))
            plot.ion()

        # Main training loop
        for train_ind in range(start_iter, cfg.TRAIN.NUM_ITERATION + 1):
            self.lr_scheduler.step()

            data_timer.tic()
            batch_img, batch_voxel = train_queue.get()
            data_timer.toc()

            if self.net.is_x_tensor4:
                batch_img = batch_img[0]

            # Apply one gradient step
            train_timer.tic()
            loss = self.train_loss(batch_img, batch_voxel)
            train_timer.toc()

            #             print(loss)
            # training_losses.append(loss.data)  转换为numpy数组
            # print(type(loss))
            # print(loss.data.numpy())
            # print(loss.data.numpy().shape)
            # print(type(loss.data.numpy()))
            if (torch.cuda.is_available()):
                training_losses.append(loss.cpu().data.numpy())
            else:
                training_losses.append(loss.data.numpy())

            # Decrease learning rate at certain points
            if train_ind in lr_steps:
                #for pytorch optimizer, learning rate can only be set when the optimizer is created
                #or using torch.optim.lr_scheduler
                print('Learing rate decreased to %f: ' %
                      cfg.TRAIN.LEARNING_RATES[str(train_ind)])

            # '''
            # Debugging modules
            # '''

            # Print status, run validation, check divergence, and save model.
            if train_ind % cfg.TRAIN.PRINT_FREQ == 0:  #40
                # Print the current loss
                print(
                    '%s Iter: %d Loss: %f' % (datetime.now(), train_ind, loss))
                '''
                @TODO(dingyadong): loss dynamic Visualization
                '''
                # plot
                if (train_ind != 0):
                    steps = np.linspace(
                        0, train_ind, train_ind + 1, dtype=np.float32)
                    if cfg.TRAIN.SHOW_LOSS:  # 要动态打印
                        plot.plot(steps, training_losses, 'b-')
                        plot.draw()
                # plot.pause(0.05)

            if train_ind % cfg.TRAIN.VALIDATION_FREQ == 0 and val_queue is not None:
                # Print test loss and params to check convergence every N iterations

                val_losses = 0
                for i in range(cfg.TRAIN.NUM_VALIDATION_ITERATIONS):
                    batch_img, batch_voxel = val_queue.get()
                    val_loss = self.train_loss(batch_img, batch_voxel)
                    val_losses += val_loss
                var_losses_mean = val_losses / cfg.TRAIN.NUM_VALIDATION_ITERATIONS
                print('%s Test loss: %f' % (datetime.now(), var_losses_mean))

            if train_ind % cfg.TRAIN.NAN_CHECK_FREQ == 0:
                # Check that the network parameters are all valid
                nan_or_max_param = max_or_nan(self.net.parameters())
                if has_nan(nan_or_max_param):
                    print('NAN detected')
                    break

            if train_ind % cfg.TRAIN.SAVE_FREQ == 0 and not train_ind == 0:
                self.save(training_losses, save_dir, train_ind)

            #loss is a Variable containing torch.FloatTensor of size 1
            if loss.data > cfg.TRAIN.LOSS_LIMIT:
                print("Cost exceeds the threshold. Stop training")
                break

        if cfg.TRAIN.SHOW_LOSS:  # 要动态打印ƒ
            plot.ioff()
            plot.show()

    def save(self, training_losses, save_dir, step):
        ''' Save the current network parameters to the save_dir and make a
        symlink to the latest param so that the training function can easily
        load the latest model'''

        save_path = os.path.join(save_dir, 'checkpoint.%d.tar' % (step))

        #both states of the network and the optimizer need to be saved
        state_dict = {'net_state': self.net.state_dict()}
        state_dict.update({'optimizer_state': self.optimizer.state_dict()})
        torch.save(state_dict, save_path)

        # Make a symlink for weights.npy
        symlink_path = os.path.join(save_dir, 'checkpoint.tar')
        if os.path.lexists(symlink_path):
            os.remove(symlink_path)

        # Make a symlink to the latest network params
        os.symlink("%s" % os.path.abspath(save_path), symlink_path)

        # Write the losses
        with open(os.path.join(save_dir, 'loss.%d.txt' % step), 'w') as f:
            f.write('\n'.join([str(l) for l in training_losses]))

    def load(self, filename):
        if os.path.isfile(filename):
            print("loading checkpoint from '{}'".format(filename))
            checkpoint = torch.load(
                filename, map_location=lambda storage, loc: storage)

            net_state = checkpoint['net_state']
            optim_state = checkpoint['optimizer_state']

            self.net.load_state_dict(net_state)
            self.optimizer.load_state_dict(optim_state)
        else:
            raise Exception("no checkpoint found at '{}'".format(filename))

    def test_output(self, x, y=None):
        """
        Generate the reconstruction, loss, and activation. Evaluate loss if
        ground truth output is given. Otherwise, return reconstruction and
        activation.
        In test mode, if y is None, then the out is the [prediction].
        In test mode, if y is not None, then the out is [prediction, loss].
        """
        x = torch.FloatTensor(x)
        if y is not None:
            y = torch.FloatTensor(y)

        if torch.cuda.is_available():
            x.cuda(async=True)
            x = x.type(torch.cuda.FloatTensor)
            if y is not None:
                y.cuda(async=True)
                y = y.type(torch.cuda.FloatTensor)

        x = Variable(x, requires_grad=False)
        if y is not None:
            y = Variable(y, requires_grad=False)

        # Parse the result
        results = self.net(x, y, test=True)
        prediction = results[0]

        activations = None
        if len(results) > 2:
            activations = results[2:]

        if y is None:  #没有loss
            return prediction, activations
        else:
            loss = results[1]
            return prediction, loss, activations

    # def graph_view(self, x):
    #     import hiddenlayer as hl
    #     x = torch.FloatTensor(x)
    #     if torch.cuda.is_available():
    #         x.cuda(async=True)
    #         x = x.type(torch.cuda.FloatTensor)
    #     x = Variable(x)
    #     hl.build_graph(self.net, (x, ))
