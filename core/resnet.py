'''
    The model architecture for resnet-50-v2, original paper: Identity Mappings in Deep Residual Networks
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import nn
import sys


class ResNet():
    def __init__(self, params):

        self.data_format_ = params.get('data_format', 'channels_first')
        self.batch_ = params.get('batch', 1)
        self.l2_weight_ = params.get('l2_weight', 0.0001)
        self.init_lr_ = params.get('init_lr', 0.01)
        self.init_filters_ = params.get('init_filters', 64)
        self.init_kernel_size_ = params.get('init_kernel_size', 7)
        self.init_conv_stride_ = params.get('init_conv_stride', 2)
        self.init_pool_size_ = params.get('init_pool_size', 3)
        self.init_pool_stride_ = params.get('init_pool_stride', 2)
        self.num_filters_ = params.get('num_filters', [64, 128, 256, 512])
        self.stage_number_ = params.get('stage_numbers', [2, 3, 4, 5])
        self.num_blocks_ = params.get('num_blocks', [3, 4, 6, 3])
        self.res_strides_ = params.get('res_strides', [1, 2, 2, 2])
        self.bn_momentum_ = params.get('bn_momentum', 0.9)
        self.bn_epsilon_ = params.get('bn_epsilon', 1e-5)

        # below are BN params of pre-trained ImageNet of ResNet-50-v2
        #_BATCH_NORM_DECAY = 0.997
        #_BATCH_NORM_EPSILON = 1e-5

        if self.data_format_ is not 'channels_first':
            sys.exit("Invalid data format. Must be 'channels_first'.")

    def res_stage(self, inputs, num_filters, stage_num, num_blocks, stride, training):
        '''
        Residual stage: aka C2, C3, C4, C5; block sizes: 3, 4, 6, 3;
        :param inputs: inputs to the stage
        :param num_filters: number of filters of the 1st residual block, the last is num_filters*4
        :param stage_num: can only be 2, 3, 4, 5; they'll be renamed as C2 ~ C5
        :param num_blocks: can only be 3, 4, 6, 3
        :param stride: can only 1 for C2 (don't down-sample in 1st res block), 2 for others (for down-sample in 1st res block)
        :param training: boolean
        :return: output of the stage
        '''

        with tf.variable_scope('C' + str(stage_num)):
            ############################ 1st block with shortcut conv ##################################
            with tf.variable_scope('block1'):
            # 1st res block with stride (for down-sampling in C3, C4, C5)
                in_dim = self.num_filters_[stage_num-2] if stage_num == 2 else self.num_filters_[stage_num-3]*4
                out_dim = num_filters
                inputs = nn.res_block(inputs=inputs, filters=[in_dim, out_dim], shortcut=True,
                                      stride=stride, l2_decay=self.l2_weight_, momentum=self.bn_momentum_,
                                      epsilon=self.bn_epsilon_, training=training, data_format=self.data_format_,
                                      first_block=True)

            ############################ the rest blocks with indentity skip ##################################
            for block_id in range(2, num_blocks+1):
                with tf.variable_scope('block' + str(block_id)):
                    inputs = nn.res_block(inputs=inputs, filters=[num_filters, num_filters], shortcut=False,
                                          stride=1, l2_decay=self.l2_weight_, momentum=self.bn_momentum_,
                                          epsilon=self.bn_epsilon_, training=training, data_format=self.data_format_,
                                          first_block=False)

        return inputs


    def build_model(self, inputs, training):
        '''
        :param inputs: input tensor
        :training: boolean
        :return: output of network
        '''

        # the backbone
        with tf.variable_scope('backbone'):
            ############################ initial conv 7x7, down-sample 4x ##################################
            inputs = nn.conv_layer(inputs=inputs, filters=[3, self.init_filters_], kernel_size=self.init_kernel_size_,
                                   stride=self.init_conv_stride_, l2_decay=self.l2_weight_, training=training,
                                   data_format=self.data_format_)
            # down-sample again by pooling
            inputs = nn.max_pool(inputs=inputs, pool_size=self.init_pool_size_, pool_stride=self.init_pool_stride_,
                                 data_format=self.data_format_)

            ############################ Resnet stages C2 ~ C5 ##################################
            for stage_id in range(2, 6):
                inputs = self.res_stage(inputs=inputs, num_filters=self.num_filters_[stage_id-2],
                                   stage_num=self.stage_number_[stage_id-2],
                                   num_blocks=self.num_blocks_[stage_id-2],
                                   stride=self.res_strides_[stage_id-2],
                                   training=training)

            ############################ ImgNet layers ##################################
            with tf.variable_scope('tail'):
                inputs = nn.batch_norm(inputs=inputs, training=training, momentum=self.bn_momentum_,
                                       epsilon=self.bn_epsilon_, data_format=self.data_format_)
                inputs = tf.nn.relu(inputs)

                axes = [2, 3]
                inputs = tf.reduce_mean(inputs, axes, keepdims=True)
                inputs = tf.squeeze(inputs, axes) # [batch, 2048]
                inputs = nn.dense_layer(inputs=inputs, weight_size=[2048, 1001], l2_decay=self.l2_weight_, training=training)

        return inputs

    def inference(self, dense_out):
        '''
        :param dense_out: a tensor with shape [batch, 1001]
        :return: a tensor with shape [batch], each vector element is predicted label id
        '''

        prob_out = tf.nn.softmax(logits=dense_out, axis=-1)
        pred_label = tf.reshape(tf.math.argmax(input=prob_out, axis=-1), [-1]) # to a vector

        return pred_label

    def loss(self, dense_out, gt_label):
        '''
        :param dense_out: output of dense layer with shape [batch, 1001]
        :param gt_label: gt label of current batch with shape [batch]
        :return: scalar lose (train loss + l2_loss)
        '''

        # l2_loss, already multiplied by decay when created graph
        l2_loss = tf.add_n(tf.get_collection('l2_losses'), name='l2_loss')
        tf.summary.scalar(name='l2_loss', tensor=l2_loss)

        # cls loss
        gt_label = tf.cast(tf.reshape(gt_label, [-1]), tf.int64)
        cross_entropy_mean = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_label, logits=dense_out))
        tf.summary.scalar(name='cls_loss', tensor=cross_entropy_mean)

        # total_loss
        total_loss = l2_loss + cross_entropy_mean
        tf.summary.scalar(name='total_loss', tensor=total_loss)

        return total_loss


