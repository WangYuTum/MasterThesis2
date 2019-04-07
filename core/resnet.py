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

    def build_templar(self, input_z, training, reuse=False):
        '''
        :param input_z: [batch, 3, 127, 127]
        :param training: boolean
        :param reuse: boolean
        :return: [15, 15, 256, batch]
        '''

        # build templar initial conv: [7x7,4,64], down-sample 2x
        with tf.variable_scope('temp_begin'):
            temp_begin_feat = nn.conv_layer(inputs=input_z, filters=[4, self.init_filters_], kernel_size=self.init_kernel_size_,
                                            stride=self.init_conv_stride_, l2_decay=self.l2_weight_, training=training,
                                            data_format=self.data_format_, pad='VALID', dilate_rate=1)

        # build backbone
        z_feat = self.build_model(input=temp_begin_feat, training=training, reuse=reuse)
        # build templar exclusive branch
        with tf.variable_scope('temp_adjust'):
            z_adjust = nn.conv_layer(inputs=z_feat, filters=[1024, 256], kernel_size=1,
                                     stride=1, l2_decay=self.l2_weight_, training=training,
                                     data_format=self.data_format_, pad='SAME', dilate_rate=1)
            z_feat = tf.transpose(z_adjust, [2, 3, 1, 0])  # reshape to [15, 15, 256, batch]

        return z_feat

    def build_search(self, input_x, training, reuse=False):
        '''
        :param input_x:  [batch, 3, 255, 255]
        :param training: boolean
        :param reuse: boolean
        :return: [batch, 256, 31, 31]
        '''

        # build search initial conv: [7x7,4,64], down-sample 2x
        with tf.variable_scope('search_begin'):
            search_begin_feat = nn.conv_layer(inputs=input_x, filters=[4, self.init_filters_], kernel_size=self.init_kernel_size_,
                                              stride=self.init_conv_stride_, l2_decay=self.l2_weight_, training=training,
                                              data_format=self.data_format_, pad='VALID', dilate_rate=1)

        # build backbone, reuse vars
        x_feat = self.build_model(input=search_begin_feat, training=training, reuse=reuse)
        # build search exclusive branch
        with tf.variable_scope('search_adjust'):
            x_adjust = nn.conv_layer(inputs=x_feat, filters=[1024, 256], kernel_size=1,
                                     stride=1, l2_decay=self.l2_weight_, training=training,
                                     data_format=self.data_format_, pad='SAME', dilate_rate=1)
            x_feat = x_adjust

        return x_feat

    def build_cc_mask(self, z_feat, x_feat, training):
        '''
        :param z_feat:  [15, 15, 256, batch]
        :param x_feat:  [batch, 256, 31, 31]
        :return: [batch, 1, 17, 17], [batch, 63*63, 17, 17]
        '''

        # do cross-correlations, batch conv operations
        with tf.variable_scope('cc_layer'):
            cc_feat = []
            for batch_i in range(int(self.batch_)):
                input_feat = x_feat[batch_i:batch_i + 1, :, :, :]  # [1, 256, 31, 31]
                input_filter = z_feat[:, :, :, batch_i:batch_i + 1]  # [15, 15, 256, 1]
                depth_cc = tf.nn.depthwise_conv2d(input=input_feat, filter=input_filter, strides=[1, 1, 1, 1],
                                                  padding='VALID', data_format='NCHW') # [1, 256, 17, 17]
                cc_feat.append(depth_cc)
            cc_out = tf.concat(axis=0, values=cc_feat)  # [batch, 256, 17, 17]
            print('Cross correlation layers built.')
        with tf.variable_scope('score_branch'):
            # conv5 score branch
            # BN + Relu + conv
            cc_out0 = nn.batch_norm(inputs=cc_out, training=training, momentum=self.bn_momentum_, epsilon=self.bn_epsilon_,
                                     data_format=self.data_format_, in_num_filters=256)
            cc_out0 = tf.nn.relu(cc_out0)
            with tf.variable_scope('conv5'):
                score = nn.conv_layer(inputs=cc_out0, filters=[256, 256], kernel_size=1,
                                      stride=1, l2_decay=self.l2_weight_, training=training,
                                      data_format=self.data_format_, pad='VALID', dilate_rate=1)
            # conv6
            with tf.variable_scope('conv6'):
                score = nn.conv_layer(inputs=score, filters=[256, 1], kernel_size=1,
                                      stride=1, l2_decay=self.l2_weight_, training=training,
                                      data_format=self.data_format_, pad='VALID', dilate_rate=1)
        with tf.variable_scope('mask_branch'):
            # conv5 mask branch
            # BN + ReLu + conv
            cc_out1 = nn.batch_norm(inputs=cc_out, training=training, momentum=self.bn_momentum_,
                                    epsilon=self.bn_epsilon_,
                                    data_format=self.data_format_, in_num_filters=256)
            cc_out1 = tf.nn.relu(cc_out1)
            with tf.variable_scope('conv5'):
                mask = nn.conv_layer(inputs=cc_out1, filters=[256, 256], kernel_size=1,
                                     stride=1, l2_decay=self.l2_weight_, training=training,
                                     data_format=self.data_format_, pad='VALID', dilate_rate=1)  # [batch, 256, 17, 17]
            # conv6
            with tf.variable_scope('conv6'):
                mask = nn.conv_layer(inputs=mask, filters=[256, 63*63], kernel_size=1,
                                      stride=1, l2_decay=self.l2_weight_, training=training,
                                      data_format=self.data_format_, pad='VALID', dilate_rate=1) # [batch, 63*63, 17, 17]

        return score, mask

    def build_model(self, input, training, reuse=False):
        '''
        Build Siamese model: build two identical models on the same device and share variables
        :param input [batch, 64, 61, 61] or [batch, 64, 125, 125]
        :param training: boolean
        :param reuse: boolean
        :return: output of network as [batch, 1024, 15, 15] or [batch, 1024, 31, 31]
        '''

        # the templar branch
        stage_out = [] # outputs of c2 ~ c4
        with tf.variable_scope('backbone', reuse=reuse):
            # down-sample 2x by pooling
            inputs = nn.max_pool(inputs=input, pool_size=self.init_pool_size_, pool_stride=self.init_pool_stride_,
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
            out_feat = stage_out[2]

        return out_feat

    def loss_score_mask(self, batch, score_map, score_gt, score_weight, mask_logits, gt_mask, gt_mask_weights,
                        alpha, bete, scope):
        '''
        :param batch: batch size per gpu, must be fixed when building graph
        :param score_map: [batch, 1, 17, 17], pred score map for each pair, tf.float32
        :param score_gt: [batch, 1, 17, 17], gt score map for each pair, tf.int32
        :param score_weight: [batch, 1, 17, 17], balanced weight for score map, tf.float32
        :param mask_logits: [batch, 63*63, 17, 17], tf.float32
        :param gt_mask: [batch, 13, 127, 127], tf.int32, binary mask for 13 positive positions
        :param gt_mask_weights: [batch, 13, 127, 127], tf.float32, balance weights for 13 positive positions
        :param alpha: loss weight for score, tf.float32
        :param beta: loss weight for mask, tf.float32
        :param scope: context of the current tower, VERY important in multi-GPU setup
        :return: alpha * score_loss + beta * mask_loss + l2_loss
        '''
        print('Building loss...')

        #########################################################
        # l2_loss, already multiplied by decay when created graph.
        #########################################################
        # Extract the l2 loss for current tower, and add to summary
        losses = tf.get_collection('l2_losses', scope=scope)
        l2_total = tf.add_n(losses)
        tf.summary.scalar(name='%s_l2' % scope, tensor=l2_total)
        print('L2 loss built.')

        ########################## Loss for score map ##############################
        score_gt = tf.cast(score_gt, tf.float32)
        neg_score_gt = score_gt - 1.0
        final_score_gt = score_gt + neg_score_gt

        # use logistic loss from paper
        a = -tf.multiply(score_map, final_score_gt)
        b = tf.nn.relu(a)
        loss = b+tf.log(tf.exp(-b)+tf.exp(a-b))
        #score_loss = tf.reduce_mean(tf.multiply(score_weight, loss))

        # use balanced cross-entropy on score maps
        score_loss = self.balanced_sigmoid_cross_entropy(logits=score_map, gt=score_gt, weight=score_weight)
        tf.summary.scalar(name='%s_score' % scope, tensor=score_loss)
        print('Score loss built.')


        ########################## Loss for Mask ##############################
        # only compute loss for positive positions
        # score_gt: [n, 1, 17, 17], tf.int32
        # mask_logits: [n, 63*63, 17, 17], tf.float32
        # gt_mask: [n, 13, 127, 127], tf.int32
        # gt_mask_weights: [n, 13, 127, 127], tf.float32

        ## get positive positions as boolean mask
        bool_mask = tf.math.equal(score_gt, 1)  # [n, 1, 17, 17], tf.bool
        bool_mask = tf.squeeze(tf.transpose(bool_mask, [0, 2, 3, 1]), -1)  # [n,17,17], tf.bool, n*13 true values
        ## extract mask_logits according to boolean mask, and upsample to 127x127
        mask_logits = tf.transpose(mask_logits, [0, 2, 3, 1])  # [n,17,17,63*63], tf.float32
        logits_posi = tf.boolean_mask(tensor=mask_logits, mask=bool_mask, name='boolean_mask_logits',
                                      axis=None)  # [n*13, 63*63]
        # upsample logits
        logits_posi = tf.reshape(logits_posi, [batch * 13, 63, 63])  # [n*13, 63, 63]
        logits_posi = tf.expand_dims(logits_posi, -1)  # [n*13, 63, 63, 1]
        logits_resized = tf.image.resize_bilinear(logits_posi, [127, 127])  # [n*13,127,127,1]
        tf.summary.image(name='mask_logits', tensor=logits_resized, max_outputs=9)
        logits_resized = tf.squeeze(logits_resized, -1)  # [n*13,127,127]
        ## prepare gt/weights
        gt_mask = tf.reshape(gt_mask, [batch * 13, 127, 127])  # [n*13, 127,127]
        # tf.summary.image(name='gt_masks', tensor=tf.cast(tf.expand_dims(gt_mask, -1)*192, tf.uint8), max_outputs=9)
        gt_mask_weights = tf.reshape(gt_mask_weights, [batch * 13, 127, 127])  # [n*13, 127,127]
        ### prepare to feed entropy: reshape -> flatten
        old_shape = tf.shape(logits_resized)
        new_shape = [old_shape[0], old_shape[1] * old_shape[2]]
        logits = tf.reshape(logits_resized, new_shape)  # [n*13, 127*127]
        gt = tf.reshape(gt_mask, new_shape)  # [n*13, 127*127]
        weight = tf.reshape(gt_mask_weights, new_shape)  # [n*13, 127*127]
        # compute loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(gt, tf.float32), logits=logits)  # [n*13, h*w]
        balanced_loss = tf.multiply(loss, weight)  # [n*13, h*w]
        balanced_loss = tf.reduce_sum(balanced_loss, axis=1)  # compute sum of weighted loss for each batch
        mask_loss = tf.reduce_mean(balanced_loss)
        tf.summary.scalar(name='%s_mask' % scope, tensor=mask_loss)
        print('Mask loss built.')


        ########################## total loss ##############################
        loss_with_mask = l2_total + alpha * score_loss + bete * mask_loss
        loss_no_mask = l2_total + alpha * score_loss
        tf.summary.scalar(name='%s_total_with_mask' % scope, tensor=loss_with_mask)
        tf.summary.scalar(name='%s_total_no_mask' % scope, tensor=loss_no_mask)

        return loss_with_mask, loss_no_mask


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


