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
    init_lr = base_lr
    warmup_lr = init_lr / 5
    boundary_epochs = [1, 2, 3, 4, 5, 25, 45, 65]
    decay_rates = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0, 0.5*5.0, 0.25*5.0, 0.05*5.0]

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs] # units in iterations
    vals = [warmup_lr * decay for decay in decay_rates]
    lr = tf.train.piecewise_constant(global_step, boundaries, vals)

    return lr

def get_adam_opt(init_lr=0.001, epsilon=1e-08, beta1=0.9, beta2=0.999):
    '''
    :param init_lr: tensorflow default value
    :param epsilon: tensorflow default value, might not be a good choice.
    :param beta1: tensorflow default value
    :param beta2: tensorflow default value
    :return: adam optimizer, learning rate
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

def average_grads(tower_grads):
    '''
    :param tower_grads: a list of grads_and_vars across all towers
        each element is a list of tuples: (grad, var)
    :return: averaged grads_and_vars
    '''

    avg_grads = []
    for grads_and_vars in zip(*tower_grads):
        # each grads_and_vars has the structure:
        # [(grad0_gpu0, var0_gpu0), (grad0_gpu1, var0_gpu1), ...(grad0_gpuN, var0_gpuN)]
        # note that var0_gpu0, var0_gpu1, ... var0_gpuN point to the same shared variable on CPU
        grads = []
        for g, _ in grads_and_vars:
            expand_g = tf.expand_dims(g, 0)
            grads.append(expand_g)

        # average grads on this variable
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis=0)

        # get this variable
        var = grads_and_vars[0][1]
        # combine this gradient and variable
        grad_and_var = (grad, var)
        avg_grads.append(grad_and_var)

    return avg_grads

def apply_lr(grads_vars, global_step, iters_per_ep):

    new_grads_vars = []
    boundary = [iters_per_ep*5]
    vals = [1e-6, 1.0]
    grad_decay = tf.train.piecewise_constant(global_step, boundary, vals)
    for var in grads_vars:
        var_name = str(var[1].name).split(':')[0]
        if var_name.find('backbone') != -1: # the backbone
            if var_name.find('backbone/beta') == -1 or var_name.find('backbone/gamma') == -1: # if not newly created vars
                grad = grad_decay * var[0]
            else:
                grad = var[0]
        else:
            grad = var[0]
        new_grads_vars.append((grad, var[1]))

    return new_grads_vars