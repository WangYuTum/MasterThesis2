'''
    Very basic building blocks for resnet-50-v4, wrapper for tf low-level functions
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow.python.training import moving_averages
import re


def get_var_cpu_with_decay(name, shape, l2_decay, initializer, training):
    '''
    :param name: name of the variable
    :param shape: shape of the variable
    :param l2_decay: float32 decay
    :param initializer:
    :param training: boolean
    :return: variable on cpu
    '''

    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32, trainable=training)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), l2_decay, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)

    return var


def get_var_cpu_no_decay(name, shape, initializer, training):
    '''
    :param name: name of the variable
    :param shape: shape of the variable
    :param initializer:
    :param training: boolean
    :return: variable on cpu
    '''

    with tf.device('/cpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32, trainable=training)

    return var

def get_var_gpu_no_decay(name, shape, initializer, training):
    '''
    :param name: name of the variable
    :param shape: shape of the variable
    :param initializer:
    :param training: boolean
    :return: variable on gpu
    '''

    with tf.device('/gpu:0'):
        var = tf.get_variable(name=name, shape=shape, initializer=initializer, dtype=tf.float32, trainable=training)

    return var

def pad_before_conv(inputs, kernel_size, data_format):
    """
        Padding scheme from tensorflow official resnet model
    """

    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv_layer(inputs, filters, kernel_size, stride, l2_decay, training, data_format):
    ''' Only do convolution, no bias, no activation
    :param inputs: input tensor
    :param filters: [in_dim, out_dim]
    :param kernel_size: single int
    :param stride:  single int
    :param l2_decay: float32 decay
    :param training: boolean
    :param data_format: either be "channels_last" or "channels_first"
    :return: outputs
    '''

    # padding
    if stride > 1:
        inputs = pad_before_conv(inputs=inputs, kernel_size=kernel_size, data_format=data_format)
    padding = 'SAME' if stride == 1 else 'VALID'
    strides = [1, 1, stride, stride] if data_format == 'channels_first' else [1, stride, stride, 1]

    data_format = 'NCHW' if data_format == 'channels_first' else 'NHWC'
    kernel = get_var_cpu_with_decay('kernel', [kernel_size, kernel_size, filters[0], filters[1]],
                                       l2_decay, tf.glorot_uniform_initializer(), training)
    print('Create {0}, {1}'.format(re.sub(':0', '', kernel.name), [kernel_size, kernel_size, filters[0], filters[1]]))
    outputs = tf.nn.conv2d(input=inputs, filter=kernel, strides=strides, padding=padding,
                           use_cudnn_on_gpu=True, data_format=data_format)

    return outputs


def max_pool(inputs, pool_size, pool_stride, data_format):
    '''
    Assuming inputs dims are even. Output dim reduced by half after pooling

    :param inputs: input tensor
    :param pool_size: single int
    :param pool_stride: single int
    :param data_format: either be "channels_last" or "channels_first"
    :return: outputs
    '''

    outputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=pool_size, strides=pool_stride, padding='same',
        data_format=data_format)

    return outputs

def dense_layer(inputs, weight_size, l2_decay, training):
    '''
    :param inputs: must have shape [N, feat_dim]
    :param weight_size: [feat_dim, out_dim]
    :param l2_decay:
    :param training: boolean
    :return: outputs
    '''

    with tf.variable_scope('dense'):
        weight_mat = get_var_cpu_with_decay(name='kernel', shape=weight_size, l2_decay=l2_decay,
                                            initializer=tf.glorot_uniform_initializer(), training=training)
        bias_vec = get_var_cpu_no_decay(name='bias', shape=weight_size[1], initializer=tf.zeros_initializer(),
                                        training=training)
        print('Create {0}, {1}'.format(weight_mat.name, weight_size))
        print('Create {0}, {1}'.format(bias_vec.name, [weight_size[1]]))
    inputs = tf.linalg.matmul(inputs, weight_mat)
    inputs = tf.add(inputs, bias_vec)
    # inputs = tf.nn.relu(inputs)

    return inputs

def res_block(inputs, filters, shortcut, stride, l2_decay, momentum, epsilon, training, data_format, first_block):
    '''
    Residual block with bottleneck
    :param inputs: input tensor
    :param filters: [in_dim, out_dim]
    :param shortcut: either comes from inputs or from side conv
    :param stride: down-sample if stride=2 (only effective on the 2nd conv)
    :param l2_decay: l2 weight for kernels/biases
    :param momentum: decay factor for BN moving statistics
    :param epsilon: for BN
    :param training: boolean
    :param data_format: either be "channels_last" or "channels_first"
    :param first_block: boolean, determine the kernel dims
    :return: outputs
    '''
    shortcut = inputs
    # bn + relu first
    with tf.variable_scope('conv1'):
        inputs = batch_norm(inputs=inputs, training=training, momentum=momentum, epsilon=epsilon, data_format=data_format)
        inputs = tf.nn.relu(inputs)

    if first_block:
        # [in_dim, out_dim] are input_dim of the 1st conv and output_dim of the 1st conv
        filters_1 = [filters[0], filters[1]]
        filters_2 = [filters[1], filters[1]]
        filters_3 = [filters[1], filters[1]*4]
        with tf.variable_scope('shortcut'):
            # shortcut conv with stride (for down-sampling in C3, C4, C5)
            shortcut = conv_layer(inputs=inputs, filters=[filters[0], filters[1]*4], kernel_size=1, stride=stride,
                                     l2_decay=l2_decay, training=training, data_format=data_format)
    else:
        # [in_dim, out_dim] are both the intermediate conv dim
        filters_1 = [filters[0]*4, filters[0]]
        filters_2 = [filters[0], filters[0]]
        filters_3 = [filters[0], filters[0]*4]

    # 1st conv
    with tf.variable_scope('conv1'):
        # inputs = batch_norm(inputs=inputs, training=training, momentum=momentum, epsilon=epsilon, data_format=data_format)
        # inputs = tf.nn.relu(inputs)
        inputs = conv_layer(inputs=inputs, filters=filters_1, kernel_size=1, stride=1, l2_decay=l2_decay, training=training,
                            data_format=data_format)

    # 2nd bn + relu + conv (down-sample if stride=2)
    with tf.variable_scope('conv2'):
        inputs = batch_norm(inputs=inputs, training=training, momentum=momentum, epsilon=epsilon, data_format=data_format)
        inputs = tf.nn.relu(inputs)
        inputs = conv_layer(inputs=inputs, filters=filters_2, kernel_size=3, stride=stride, l2_decay=l2_decay, training=training,
                            data_format=data_format)

    # 3rd bn + relu + conv
    with tf.variable_scope('conv3'):
        inputs = batch_norm(inputs=inputs, training=training, momentum=momentum, epsilon=epsilon, data_format=data_format)
        inputs = tf.nn.relu(inputs)
        inputs = conv_layer(inputs=inputs, filters=filters_3, kernel_size=1, stride=1, l2_decay=l2_decay, training=training,
                            data_format=data_format)

    # fuse with shortcut
    outputs = inputs + shortcut

    return outputs


def batch_norm(inputs, training, momentum, epsilon, data_format):
    '''
    :param inputs: input tensor
    :param training: boolean
    :param momentum: decay factor for BN moving statistics
    :param epsilon: for BN
    :param data_format: either be "channels_last" or "channels_first"
    :return: normalized inputs
    '''

    if data_format is not 'channels_first':
        print('Data format not channels_first in BN')
        sys.exit(-1)
    num_filters = inputs.get_shape().as_list()[1]
    new_shape = [1, num_filters, 1, 1]

    ###############################  Single GPU Impl.  ##################################
    # during inference, we use final moving_mean, moving_var (which MUST be initialised from checkpoint)
    if not training:
        moving_mean = get_var_cpu_no_decay(name='moving_mean', shape=[num_filters], initializer=tf.constant_initializer(),
                                           training=False)
        print('Create {0}, {1}'.format(re.sub(':0', '', moving_mean.name), [num_filters]))
        moving_mean = tf.reshape(moving_mean, new_shape) # tf.nn.batch_normalization() requires full shape
        moving_variance = get_var_cpu_no_decay(name='moving_variance', shape=[num_filters], initializer=tf.constant_initializer(1),
                                           training=False)
        print('Create {0}, {1}'.format(re.sub(':0', '', moving_variance.name), [num_filters]))
        moving_variance = tf.reshape(moving_variance, new_shape)
        beta = get_var_cpu_no_decay(name='beta', shape=[num_filters], initializer=tf.zeros_initializer(), training=False)
        print('Create {0}, {1}'.format(re.sub(':0', '', beta.name), [num_filters]))
        beta = tf.reshape(beta, new_shape)
        gamma = get_var_cpu_no_decay(name='gamma', shape=[num_filters], initializer=tf.ones_initializer(), training=False)
        print('Create {0}, {1}'.format(re.sub(':0', '', gamma.name), [num_filters]))
        gamma = tf.reshape(gamma, new_shape)

        # normalize the inputs using global/moving statistics
        inputs = tf.nn.batch_normalization(x=inputs, mean=moving_mean, variance=moving_variance, offset=beta,
                                           scale=gamma, variance_epsilon=epsilon)
    else:
        # we must do the following:
        #   * make beta, gamma as trainable parameters on cpu:0
        #   * make moving_mean, moving_var as non-trainable parameters on gpu:0
        #   * load moving_mean, moving_variance, beta, gamma if there's pre-trained checkpoint
        #   * define ops to compute local batch mean, var (they are used to normalize the inputs)
        #   * define ops to compute moving_mean, moving_var (they only need to be updated for each iteration, and used during inference)
        moving_mean = get_var_gpu_no_decay(name='moving_mean', shape=[num_filters], initializer=tf.constant_initializer(),
                                           training=False)
        print('Create {0}, {1}'.format(re.sub(':0', '', moving_mean.name), [num_filters]))
        moving_variance = get_var_gpu_no_decay(name='moving_variance', shape=[num_filters], initializer=tf.constant_initializer(1),
                                               training=False)
        print('Create {0}, {1}'.format(re.sub(':0', '', moving_variance.name), [num_filters]))
        beta = get_var_cpu_no_decay(name='beta', shape=[num_filters], initializer=tf.zeros_initializer(), training=True)
        print('Create {0}, {1}'.format(re.sub(':0', '', beta.name), [num_filters]))
        gamma = get_var_cpu_no_decay(name='gamma', shape=[num_filters], initializer=tf.ones_initializer(), training=True)
        print('Create {0}, {1}'.format(re.sub(':0', '', gamma.name), [num_filters]))
        # compute local batch mean, var and update moving_mean, moving_var
        batch_mean, batch_variance = tf.nn.moments(inputs, axes=[0, 2, 3], keepdims=False) # produces two scalars
        update_mean_op = moving_averages.assign_moving_average(moving_mean, batch_mean, momentum)
        update_var_op = moving_averages.assign_moving_average(moving_variance, batch_variance, momentum)
        # in this case, moving statistic will be updated here before the actual batch norm execution
        with tf.control_dependencies([update_mean_op, update_var_op]):
            inputs = tf.identity(inputs)

        # normalize the inputs using local batch statistics
        batch_mean = tf.reshape(batch_mean, new_shape)
        batch_variance = tf.reshape(batch_variance, new_shape)
        beta = tf.reshape(beta, new_shape)
        gamma = tf.reshape(gamma, new_shape)
        inputs = tf.nn.batch_normalization(x=inputs, mean=batch_mean, variance=batch_variance, offset=beta,
                                           scale=gamma, variance_epsilon=epsilon)

    ###############################  Multi-GPU Impl.  ##################################

    return inputs

