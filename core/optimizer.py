'''
 A helper script to construct an optimizer with given parameters
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def lr_scheduler(base_lr, batches_per_epoch, batch_size, global_step, bnorm):
    '''
    :param batches_per_epoch:
    :param global_step:
    :return:
    '''
    init_lr = base_lr * batch_size / bnorm # if actuall batch_size < 512, the lr is reduced
    boundary_epochs = [30, 60, 80, 90]
    decay_rates = [1, 0.1, 0.01, 0.001, 1e-4]

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs] # units in iterations
    vals = [init_lr * decay for decay in decay_rates]
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)

    return lr

def get_adam_opt(init_lr=0.001, epsilon=1e-08, beta1=0.9, beta2=0.999):
    '''
    :param init_lr: tensorflow default value
    :param epsilon: tensorflow default value, might not be a good choice.
    :param beta1: tensorflow default value
    :param beta2: tensorflow default value
    :return: adam optimizer
    '''

    opt = tf.train.AdamOptimizer(learning_rate=init_lr, epsilon=epsilon,
                                 beta1=beta1, beta2=beta2)

    return opt

def get_momentum_opt(base_lr, batches_per_epoch, global_step, batch_size=512, momentum=0.9, bnorm=512):
    '''
    :param base_lr: base_lr, might be reduced automatically if batch_size is smaller than 512
    :param momentum: default to 0.9, a generally good choice
    :return: momentum optimizer
    '''

    lr = lr_scheduler(base_lr, batches_per_epoch, batch_size, global_step, bnorm)
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)

    return opt, lr