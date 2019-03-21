'''
    The model architecture for resnet-50-v2, original paper: Identity Mappings in Deep Residual Networks
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from . import nn
import sys


class ResNetSiam():
    def __init__(self, params):

        self.data_format_ = params.get('data_format', 'channels_first')
        self.batch_ = params.get('batch', 8)
        self.l2_weight_ = params.get('l2_weight', 0.0001)
        self.init_lr_ = params.get('init_lr', 0.001)
        self.init_filters_ = params.get('init_filters', 64)
        self.init_kernel_size_ = params.get('init_kernel_size', 7)
        self.init_conv_stride_ = params.get('init_conv_stride', 2)
        self.init_pool_size_ = params.get('init_pool_size', 3)
        self.init_pool_stride_ = params.get('init_pool_stride', 2)
        self.num_filters_ = params.get('num_filters', [64, 128, 256, 512])
        self.stage_number_ = params.get('stage_numbers', [2, 3, 4, 5])
        self.num_blocks_ = params.get('num_blocks', [3, 4, 6, 3])
        self.res_strides_ = params.get('res_strides', [1, 2, 1, 1])
        self.bn_momentum_ = params.get('bn_momentum', 0.997)
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
            # 1st res block with stride 2 (if in C3, for downsampling)
            # only use dilate=2, 4 for c4, c5
                dilate_rate = 1
                if stage_num == 4:
                    dilate_rate = 2
                if stage_num == 5:
                    dilate_rate = 4
                in_dim = self.num_filters_[stage_num-2] if stage_num == 2 else self.num_filters_[stage_num-3]*4
                out_dim = num_filters
                inputs = nn.res_block(inputs=inputs, filters=[in_dim, out_dim], shortcut=True,
                                      stride=stride, l2_decay=self.l2_weight_, momentum=self.bn_momentum_,
                                      epsilon=self.bn_epsilon_, training=training, data_format=self.data_format_,
                                      first_block=True, dilate=dilate_rate)

            ############################ the rest blocks with indentity skip ##################################
            for block_id in range(2, num_blocks+1):
                with tf.variable_scope('block' + str(block_id)):
                    inputs = nn.res_block(inputs=inputs, filters=[num_filters, num_filters], shortcut=False,
                                          stride=1, l2_decay=self.l2_weight_, momentum=self.bn_momentum_,
                                          epsilon=self.bn_epsilon_, training=training, data_format=self.data_format_,
                                          first_block=False, dilate=1)

        return inputs

    def build_inf_model(self, inputs):
        '''
        :param inputs: [n, 3, h, w]
        :return: [n, 256, h/4, w/4] feature map
        '''

        training = False
        # the backbone
        stage_out = [] # outputs of c2 ~ c5
        with tf.variable_scope('backbone'):
            ############################ initial conv 7x7, down-sample 4x ##################################
            inputs = nn.conv_layer(inputs=inputs, filters=[3, self.init_filters_], kernel_size=self.init_kernel_size_,
                                   stride=self.init_conv_stride_, l2_decay=self.l2_weight_, training=training,
                                   data_format=self.data_format_, pad='SAME', dilate_rate=1)
            # down-sample again by pooling
            inputs = nn.max_pool(inputs=inputs, pool_size=self.init_pool_size_, pool_stride=self.init_pool_stride_,
                                 data_format=self.data_format_, pad='VALID')

            ############################ Resnet stages C2 ~ C4 ##################################
            for stage_id in range(2, 5):
                inputs = self.res_stage(inputs=inputs, num_filters=self.num_filters_[stage_id-2],
                                   stage_num=self.stage_number_[stage_id-2],
                                   num_blocks=self.num_blocks_[stage_id-2],
                                   stride=self.res_strides_[stage_id-2],
                                   training=training)
                inputs = tf.identity(inputs, name='C%d_out'%stage_id)
                stage_out.append(inputs)

        ############################ Loc & Mask ##################################
        with tf.variable_scope('heads'):
            # during inf, each templar is [n, 256, 15, 15] from C4, search is [n, 256, 31, 31] from C4
            ################################## cross-correlation branch #################################
            # BN + relu first
            head_out = nn.batch_norm(inputs=stage_out[2], training=training, momentum=self.bn_momentum_,
                                    epsilon=self.bn_epsilon_,
                                    data_format=self.data_format_)
            feat_map = tf.nn.relu(head_out)

        return feat_map

    def build_templar(self, input_z, training, reuse=False):
        '''
        :param input_z: [batch, 3, 127, 127]
        :param training: boolean
        :param reuse: boolean
        :return: [15, 15, 256, batch]
        '''

        # build backbone
        z_feat = self.build_model(input=input_z, training=training, reuse=reuse)
        # build templar exclusive branch
        with tf.variable_scope('temp_adjust'):
            z_adjust = nn.conv_layer(inputs=z_feat, filters=[1024, 256], kernel_size=1,
                                     stride=1, l2_decay=self.l2_weight_, training=training,
                                     data_format=self.data_format_, pad='SAME', dilate_rate=1)
            bias_z = nn.get_var_cpu_no_decay(name='bias', shape=256, initializer=tf.zeros_initializer(),
                                             training=training)
            print('Create {0}, {1}'.format(bias_z.name, [256]))
            z_adjust = tf.nn.bias_add(value=z_adjust, bias=bias_z,
                                      data_format='NCHW')  # [batch, 256, 15, 15]
            z_feat = tf.transpose(z_adjust, [2, 3, 1, 0])  # reshape to [15, 15, 256, batch]

        return z_feat

    def build_search(self, input_x, training, reuse=False):
        '''
        :param input_x:  [batch, 3, 255, 255]
        :param training: boolean
        :param reuse: boolean
        :return: [batch, 256, 31, 31]
        '''

        # build backbone, reuse vars
        x_feat = self.build_model(input=input_x, training=training, reuse=reuse)
        # build search exclusive branch
        with tf.variable_scope('search_adjust'):
            x_adjust = nn.conv_layer(inputs=x_feat, filters=[1024, 256], kernel_size=1,
                                     stride=1, l2_decay=self.l2_weight_, training=training,
                                     data_format=self.data_format_, pad='SAME', dilate_rate=1)
            bias_x = nn.get_var_cpu_no_decay(name='bias', shape=256, initializer=tf.zeros_initializer(),
                                             training=training)
            print('Create {0}, {1}'.format(bias_x.name, [256]))
            x_feat = tf.nn.bias_add(value=x_adjust, bias=bias_x, data_format='NCHW')  # [batch, 256, 31, 31]

        return x_feat

    def build_CC(self, z_feat, x_feat, training):
        '''
        :param z_feat:  [15, 15, 256, batch]
        :param x_feat:  [batch, 256, 31, 31]
        :return: [batch, 1, 17, 17]
        '''

        # do cross-correlations, batch conv operations
        with tf.variable_scope('cc_layer'):
            score_maps = []
            for batch_i in range(int(self.batch_)):
                input_feat = x_feat[batch_i:batch_i + 1, :, :, :]  # [1, 256, 31, 31]
                input_filter = z_feat[:, :, :, batch_i:batch_i + 1]  # [15, 15, 256, 1]
                score_map = tf.nn.conv2d(input=input_feat, filter=input_filter, strides=[1, 1, 1, 1], padding='VALID',
                                         use_cudnn_on_gpu=True, data_format='NCHW')  # [1, 1, 17, 17]
                score_maps.append(score_map)
            final_scores = tf.concat(axis=0, values=score_maps)  # [batch, 1, 17, 17]
            cc_bias = nn.get_var_cpu_no_decay(name='bias', shape=1, initializer=tf.zeros_initializer(),
                                                 training=training)  # [1]
            print('Create {0}, {1}'.format(cc_bias.name, [1]))
            cc_scores = tf.nn.bias_add(value=final_scores, bias=cc_bias,
                                       data_format='NCHW')  # [batch, 1, 17, 17]
        print('Cross correlation layers built.')

        return cc_scores

    def build_model(self, input, training, reuse=False):
        '''
        Build Siamese model: build two identical models on the same device and share variables
        :param input [batch, 3, 127, 127] or [batch, 3, 255, 255]
        :param training: boolean
        :param reuse: boolean
        :return: output of network as [batch, 1024, 15, 15] or [batch, 1024, 31, 31]
        '''

        # the templar branch
        stage_out = [] # outputs of c2 ~ c4
        with tf.variable_scope('backbone', reuse=reuse):
            ############################ initial conv 7x7, down-sample 4x ##################################
            inputs = nn.conv_layer(inputs=input, filters=[3, self.init_filters_], kernel_size=self.init_kernel_size_,
                                   stride=self.init_conv_stride_, l2_decay=self.l2_weight_, training=training,
                                   data_format=self.data_format_, pad='VALID', dilate_rate=1)
            # down-sample again by pooling
            inputs = nn.max_pool(inputs=inputs, pool_size=self.init_pool_size_, pool_stride=self.init_pool_stride_,
                                 data_format=self.data_format_, pad='SAME')

            ############################ Resnet stages C2 ~ C4 ##################################
            for stage_id in range(2, 5):
                inputs = self.res_stage(inputs=inputs, num_filters=self.num_filters_[stage_id-2],
                                   stage_num=self.stage_number_[stage_id-2],
                                   num_blocks=self.num_blocks_[stage_id-2],
                                   stride=self.res_strides_[stage_id-2],
                                   training=training)
                inputs = tf.identity(inputs, name='C%d_out'%stage_id)
                stage_out.append(inputs)
            # BN + relu
            out_feat = nn.batch_norm(inputs=stage_out[2], training=training, momentum=self.bn_momentum_, epsilon=self.bn_epsilon_,
                                data_format=self.data_format_)
            out_feat = tf.nn.relu(out_feat)

        return out_feat

    def loss_score(self, score_map, score_gt, score_weight, scope):
        '''
        :param score_map: [batch/2, 1, 17, 17], pred score map for each pair, tf.float32
        :param score_gt: [batch/2, 1, 17, 17], gt score map for each pair, tf.int32
        :param score_weight: [batch/2, 1, 17, 17], balanced weight for score map, tf.float32
        :param scope: context of the current tower, VERY important in multi-GPU setup
        :return: score_loss + l2_loss
        '''
        print('Building loss...')

        #########################################################
        # l2_loss, already multiplied by decay when created graph.
        #########################################################
        # Extract the l2 loss for current tower, and add to summary
        losses = tf.get_collection('l2_losses', scope=scope)
        l2_total = tf.add_n(losses)
        tf.summary.scalar(name='%s_l2' % scope, tensor=l2_total)

        ########################## Loss for score map ##############################
        score_gt = tf.cast(score_gt, tf.float32)
        neg_score_gt = score_gt - 1.0
        final_score_gt = score_gt + neg_score_gt

        # use logistic loss from paper
        a = -tf.multiply(score_map, final_score_gt)
        b = tf.nn.relu(a)
        loss = b+tf.log(tf.exp(-b)+tf.exp(a-b))
        score_loss = tf.reduce_mean(tf.multiply(score_weight, loss))

        # use balanced cross-entropy on score maps
        #score_loss = self.balanced_sigmoid_cross_entropy(logits=score_map, gt=score_gt, weight=score_weight)
        tf.summary.scalar(name='%s_score' % scope, tensor=score_loss)

        ########################## total loss ##############################
        total_loss = l2_total + score_loss
        tf.summary.scalar(name='%s_total_loss' % scope, tensor=total_loss)

        return total_loss


    def loss(self, score_map, score_gt, score_weight, lambda_score, mask_map, mask_gt, mask_weight, lambda_mask, scope):
        '''
        :param score_map: [batch/2, 1, 33, 33], pred score map for each pair
        :param score_gt: [batch/2, 1, 33, 33], gt score map for each pair
        :param score_weight: [batch/2, 1, 33, 33], balanced weight for score map
        :param lambda_score: tf.float32, loss weight for score
        :param mask_map: [16 * batch/2, 2, 32, 32], each pair has 16 seg_mask
        :param mask_gt: [16 * batch/2, 1, 32, 32], each pair has 16 gt seg_mask
        :param mask_weight: [16 * batch/2, 1, 32, 32], balanced weight for seg_mask
        :param lambda_mask: tf.float32, loss weight for mask
        :param scope: context of the current tower, VERY important in multi-GPU setup
        :return: total loss
        '''

        print('Building loss...')

        #########################################################
        # l2_loss, already multiplied by decay when created graph.
        #########################################################
        # Extract the l2 loss for current tower, and add to summary
        losses = tf.get_collection('l2_losses', scope=scope)
        l2_total = tf.add_n(losses)
        tf.summary.scalar(name='%s_l2' % scope, tensor=l2_total)


        ########################## Loss for score map ##############################
        # use balanced cross-entropy on score maps
        score_loss = self.balanced_sigmoid_cross_entropy(logits=score_map, gt=score_gt, weight=score_weight)
        tf.summary.scalar(name='%s_score' % scope, tensor=score_loss)


        ########################## Loss for segmentation ##############################
        mask_loss = self.balanced_softmax_cross_entropy(logits=mask_map, gt=mask_gt, weight=mask_weight)
        tf.summary.scalar(name='%s_mask' % scope, tensor=mask_loss)


        ########################## total loss ##############################
        total_loss = l2_total + lambda_score * score_loss + lambda_mask * mask_loss
        tf.summary.scalar(name='%s_total_loss' % scope, tensor=total_loss)

        return total_loss


    def balanced_sigmoid_cross_entropy(self, logits, gt, weight):
        '''
        :param logits: [batch, 1, h, w], tf.float32
        :param gt: [batch, 1, h, w], tf.int32
        :param weight: [batch, 1, h, w], balanced weight, tf.float32
        :return: mean loss
        '''

        # to [n, h, w, c]
        logits = tf.transpose(logits, [0, 2, 3, 1])
        gt = tf.transpose(gt, [0, 2, 3, 1])
        weight = tf.transpose(weight, [0, 2, 3, 1])
        # reshape -> flatten
        old_shape = tf.shape(logits)
        new_shape = [old_shape[0], old_shape[1]*old_shape[2]]
        logits = tf.reshape(logits, new_shape)
        gt = tf.reshape(gt, new_shape)
        weight = tf.reshape(weight, new_shape)

        # compute loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt, logits=logits) # [batch, h*w]
        balanced_loss = tf.multiply(loss, weight) # [batch, h*w]
        balanced_loss = tf.reduce_sum(balanced_loss, axis=1) # compute sum of weighted loss for each batch

        return tf.reduce_mean(balanced_loss)

    def balanced_softmax_cross_entropy(self, logits, gt, weight):
        '''
        :param logits: [batch, C, h, w], C is number of classes (=2 if binary segmentation)
        :param gt: [batch, 1, h, w]
        :param weight: [batch, 1, h, w], balanced weight
        :return: mean loss
        '''

        # to [n, h, w, c]
        logits = tf.transpose(logits, [0, 2, 3, 1])
        gt = tf.transpose(gt, [0, 2, 3, 1])
        weight = tf.transpose(weight, [0, 2, 3, 1])
        # reshape -> flatten
        old_shape = tf.shape(logits)
        logits = tf.reshape(logits, [old_shape[0], old_shape[1] * old_shape[2], old_shape[3]]) # [batch, h*w, c=2]
        gt = tf.reshape(gt, [old_shape[0], old_shape[1] * old_shape[2]]) # [batch, h*w, 1]
        weight = tf.reshape(weight, [old_shape[0], old_shape[1] * old_shape[2]]) # [batch, h*w, 1]

        # compute loss
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=logits)
        balanced_loss = tf.multiply(loss, weight)  # [batch, h*w]

        return tf.reduce_mean(balanced_loss)


    def inference(self, dense_out):
        '''
        :param dense_out: a tensor with shape [batch, 1001]
        :return: a tensor with shape [batch], each vector element is predicted label id
        '''

        prob_out = tf.nn.softmax(logits=dense_out, axis=-1)
        pred_label = tf.reshape(tf.math.argmax(input=prob_out, axis=-1), [-1]) # to a vector

        return pred_label


