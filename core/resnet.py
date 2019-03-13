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

    def build_model(self, inputs, training):
        '''
        :param inputs:
            templar images: [0:batch/2, c, h, w]
            search images: [batch/2:batch, c, h, w]
        :training: boolean
        :return: output of network
        '''

        # the backbone
        stage_out = [] # outputs of c2 ~ c5
        pyramid_inter = [] # intermediate pyramid outputs
        pyramid_out = [] # outputs of p2 ~ p5
        with tf.variable_scope('backbone'):
            ############################ initial conv 7x7, down-sample 4x ##################################
            inputs = nn.conv_layer(inputs=inputs, filters=[3, self.init_filters_], kernel_size=self.init_kernel_size_,
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

            ############################ Feature Pyramids ##################################
            """
            with tf.variable_scope('P5'):
                with tf.variable_scope('feat_down'):
                    output = nn.conv_layer(inputs=stage_out[3], filters=[2048, 256], kernel_size=1, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                pyramid_inter.append(output)
                pyramid_out.append(output)
            with tf.variable_scope('P4'):
                # 1x1 conv
                with tf.variable_scope('feat_down'):
                    output = nn.conv_layer(inputs=stage_out[2], filters=[1024, 256], kernel_size=1, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                up_size = [tf.shape(output)[2], tf.shape(output)[3]]
                # feature add
                output = tf.transpose(output, [0,2,3,1]) + tf.image.resize_images(images=tf.transpose(pyramid_inter[0], [0,2,3,1]), size=up_size)
                output = tf.transpose(output, [0, 3, 1, 2])
                pyramid_inter.append(output)
                # feature fuse
                with tf.variable_scope('feat_fuse'):
                    output = nn.conv_layer(inputs=output, filters=[256, 256], kernel_size=3, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                pyramid_out.append(output)
            with tf.variable_scope('P3'):
                # 1x1 conv
                with tf.variable_scope('feat_down'):
                    output = nn.conv_layer(inputs=stage_out[1], filters=[512, 256], kernel_size=1, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                up_size = [tf.shape(output)[2], tf.shape(output)[3]]
                # feature add
                output = tf.transpose(output, [0,2,3,1]) + tf.image.resize_images(images=tf.transpose(pyramid_inter[1], [0, 2, 3, 1]), size=up_size)
                output = tf.transpose(output, [0, 3, 1, 2])
                pyramid_inter.append(output)
                # feature fuse
                with tf.variable_scope('feat_fuse'):
                    output = nn.conv_layer(inputs=output, filters=[256, 256], kernel_size=3, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                pyramid_out.append(output)
            with tf.variable_scope('P2'):
                # 1x1 conv
                with tf.variable_scope('feat_down'):
                    output = nn.conv_layer(inputs=stage_out[0], filters=[256, 256], kernel_size=1, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                up_size = [tf.shape(output)[2], tf.shape(output)[3]]
                # feature add
                output = tf.transpose(output, [0,2,3,1]) + tf.image.resize_images(images=tf.transpose(pyramid_inter[2], [0, 2, 3, 1]), size=up_size)
                output = tf.transpose(output, [0, 3, 1, 2])
                pyramid_inter.append(output)
                # feature fuse
                with tf.variable_scope('feat_fuse'):
                    output = nn.conv_layer(inputs=output, filters=[256, 256], kernel_size=3, stride=1,
                                            l2_decay=self.l2_weight_, training=training, data_format=self.data_format_)
                pyramid_out.append(output)
            pyramid_inter.reverse() # to the order of m2, m3, m4, m5
            pyramid_out.reverse() # to the order of p2, p3, p4, p5
            """
        ############################ Loc & Mask layer ##################################
        with tf.variable_scope('heads'):
            # during training, each templar/search image have the same shape. [1, 256, 31, 31] from C4
            # the batch_size = num_train_pairs x 2, [num_pair x 2, 256, 31, 31]
            # templars are [0 : batch_size/2, 256, 31, 31], search images are [batch_size/2 : batch_size, 256, 31, 31]

            ################################## cross-correlation branch #################################
            # BN + relu first
            head_out = nn.batch_norm(inputs=stage_out[2], training=training, momentum=self.bn_momentum_, epsilon=self.bn_epsilon_,
                                data_format=self.data_format_)
            head_out = tf.nn.relu(head_out)
            # central crop the templar features to 32x32, and make a set of filters from them
            templar_feat = head_out[0:int(self.batch_/2),:,:,:] # get templar feature maps
            templar_feat = tf.transpose(templar_feat, [0,2,3,1]) # to [n, h, w, c]
            templar_feat = tf.image.crop_to_bounding_box(image=templar_feat, offset_height=8, offset_width=8,
                                                         target_height=15, target_width=15)  # [batch/2, 31, 31, 256] -> [batch/2, 15, 15, 256]
            templar_feat = tf.transpose(templar_feat, [0, 3, 1, 2]) # [n, c, h, w]
            with tf.variable_scope('temp_adjust'):
                templar_adjust = nn.conv_layer(inputs=templar_feat, filters=[1024, 256], kernel_size=1,
                                               stride=1, l2_decay=self.l2_weight_, training=training,
                                               data_format=self.data_format_, pad='SAME', dilate_rate=1)
                bias_temp = nn.get_var_cpu_no_decay(name='bias', shape=256, initializer=tf.zeros_initializer(),
                                              training=training)  # [256]
                print('Create {0}, {1}'.format(bias_temp.name, [256]))
                templar_adjust = tf.nn.bias_add(value=templar_adjust, bias=bias_temp,data_format='NCHW') # [16*batch/2, 256, 15, 15]
                templar_adjust = tf.transpose(templar_adjust, [2, 3, 1, 0])  # reshape to [15, 15, 256, batch/2]
            # extract search image feature maps from C5
            search_feat = head_out[int(self.batch_/2):self.batch_,:,:,:] # [batch/2, 256, 31, 31]
            with tf.variable_scope('search_adjust'):
                search_adjust = nn.conv_layer(inputs=search_feat, filters=[1024, 256], kernel_size=1,
                                               stride=1, l2_decay=self.l2_weight_, training=training,
                                               data_format=self.data_format_, pad='SAME', dilate_rate=1)
                bias_search = nn.get_var_cpu_no_decay(name='bias', shape=256, initializer=tf.zeros_initializer(),
                                                    training=training)  # [256]
                print('Create {0}, {1}'.format(bias_search.name, [256]))
                search_adjust = tf.nn.bias_add(value=search_adjust, bias=bias_search, data_format='NCHW') # [16*batch/2, 256, 31, 31]
            # do cross-correlations, batch/2 conv operations
            score_maps = []
            for batch_i in range(int(self.batch_/2)):
                input_feat = search_adjust[batch_i:batch_i+1, :, :, :] # [1, 256, 31, 31]
                input_filter = templar_adjust[:, :, :, batch_i:batch_i+1] # [15, 15, 256, 1]
                score_map = tf.nn.conv2d(input=input_feat, filter=input_filter, strides=[1,1,1,1], padding='VALID',
                           use_cudnn_on_gpu=True, data_format='NCHW') # [1, 1, 17, 17]
                score_maps.append(score_map)
            final_scores = tf.concat(axis=0, values=score_maps) # [batch/2, 1, 17, 17]
            with tf.variable_scope('final_bias'):
                final_bias = nn.get_var_cpu_no_decay(name='bias', shape=1, initializer=tf.zeros_initializer(),
                                                      training=training)  # [1]
                print('Create {0}, {1}'.format(final_bias.name, [1]))
                final_scores = tf.nn.bias_add(value=final_scores, bias=final_bias,
                                               data_format='NCHW')  # [batch/2, 1, 17, 17]
            print('Cross correlation layers built.')

            ################################## mask-propagte branch #################################
            # TODO: take the templar's mask, provided from data pipeline
            #templar_masks = 0 # [batch/2, 1, 32, 32]
            #search_crops = [] # [batch/2, 16, 2]: batch/2 pairs, each pair has 16 crops, each crop has [h,w]
            # crop feature maps from search feature maps, coordinates provided from data pipeline
            # reuse the var above: templar_feat [32, 32, 256, batch/2], search_feat [batch/2, 256, 64, 64]
            #concatenated_features = []
            #for batch_i in range(int(self.batch_/2)):
            #    batch_i_search_feat = search_feat[batch_i:batch_i+1, :, :, :] # [1, 256, 64, 64]
            #    batch_i_templar_feat = templar_feat[:, :, :, batch_i:batch_i+1] # [32, 32, 256, 1]
            #    batch_i_templar_mask = templar_masks[batch_i:batch_i+1, :, :, :] # [1, 1, 32, 32]
            #    batch_i_fuses = []
            #    for crop_i in range(16):
            #        crop_i_coord = search_crops[batch_i, crop_i, :] # [h, w]
            #        cropped_feat = batch_i_search_feat[:, :, crop_i_coord[0]:crop_i_coord[0]+32, crop_i_coord[1]:crop_i_coord[1]+32] # [1, 256, 32, 32]
                    # stack them
            #        stack_crop_i = tf.concat(axis=0, values=[tf.transpose(batch_i_templar_feat, [3, 2, 0, 1]),
            #                                                 batch_i_templar_mask, cropped_feat]) # [1, 513, 32, 32]
            #        batch_i_fuses.append(stack_crop_i)
                # stack for this batch_i
            #    batch_i_fuses = tf.concat(axis=0, values=batch_i_fuses) # [16, 513, 32, 32]
                # save for this batch_i
            #    concatenated_features.append(batch_i_fuses)
            # stack for all batches
            #concatenated_features = tf.concat(axis=0, values=concatenated_features) # [16*batch/2, 513, 32, 32]

            # feed to mask propagation convs
            #with tf.variable_scope('mask_prop'):
            #    mask_out = nn.mask_prop_layer(inputs=concatenated_features, training=training, l2_decay=self.l2_weight_,
            #                                  momentum=self.bn_momentum_, epsilon=self.bn_epsilon_, data_format=self.data_format_)
                # mask_out has shape [16 * batch/2, 2, 32, 32]

        return final_scores
        #return final_scores, mask_out

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

        # use loss from paper
        #a = -tf.multiply(score_map, final_score_gt)
        #b = tf.nn.relu(a)
        #loss = b+tf.log(tf.exp(-b)+tf.exp(a-b))
        #score_loss = tf.reduce_mean(tf.multiply(score_weight, loss))

        # use balanced cross-entropy on score maps
        score_loss = self.balanced_sigmoid_cross_entropy(logits=score_map, gt=score_gt, weight=score_weight)
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


