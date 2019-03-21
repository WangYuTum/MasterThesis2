'''
    The script implements a tracker by building an inference computational graph of the network. The tracker must be
    initialised by passing the first frame's rgb image and ground truth bounding box position. The track will use the
    first image/bbox to build a tf.session, and keep the target template in the main memory.
    After the initialisation, the tracker will receive one frame at a time and localize the target's position by returning
    a bbox coordinates.
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
sys.path.append('..')
from core import resnet, nn
import numpy as np

class Tracker():
    def __init__(self, num_templars, chkp):

        # build graph and ops once for all
        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94
        self._CHANNEL_MEANS = [self._R_MEAN, self._G_MEAN, self._B_MEAN]
        self._response_up = 16
        self._num_templars = num_templars
        self._pre_locations = [] # np array, # store the previous bbox positions as a list, each ele is a list: [xmin, ymin, xmax, ymax]
        self._scale_templars = [] # store the scale factor for each templar as tensor
        self._scale_templars_val = None # save values of templar scale as np array
        self._templar_img = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='templars_in')
        self._templar_bbox = tf.placeholder(dtype=tf.int32, shape=[self._num_templars, 4], name='templars_bbox_in') # for N box, [xmin, ymin, xmax, ymax]
        self._in_is_templar = tf.placeholder(dtype=tf.bool, shape=[], name='input_bool')  # true if input is templar batches
        self._search_img = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='search_in')
        self._search_bbox = tf.placeholder(dtype=tf.int32, shape=[self._num_templars, 4], name='search_bbox_in') # for N box, [xmin, ymin, xmax, ymax]

        # normalize image mean
        self._templar_img = self.mean_image_subtraction(tf.squeeze(self._templar_img, 0), self._CHANNEL_MEANS, 3)
        self._templar_img = tf.expand_dims(self._templar_img, 0)
        self._search_img = self.mean_image_subtraction(tf.squeeze(self._search_img, 0), self._CHANNEL_MEANS, 3)
        self._search_img = tf.expand_dims(self._search_img, 0)

        # crop templar image to multiple patches according to given bbox, and rescale to [127, 127]
        templar_imgs, self._scale_templars = self.crop_templars(self._templar_img, self._templar_bbox) # in [1, h, w, 3], out [n, 127, 127, 3]
        tf.summary.image(name='templar_patch', tensor=templar_imgs)

        # crop search image to multiple patches according to previous bbox predictions
        search_imgs = self.crop_searches(self._search_img, self._search_bbox) # in [1, h, w, 3], out [n, 255, 255, 3]
        tf.summary.image(name='search_patch', tensor=search_imgs)


        # build a network that takes input as image(s), and outputs feature map(s)
        def return_temp(): return templar_imgs
        def return_search(): return search_imgs
        siam_input = tf.cond(self._in_is_templar, return_temp, return_search)
        siam_model = resnet.ResNet(params={})
        # input is [n, 127, 127, 3] if templars, or [n, 255, 255, 3] if searchs
        siam_input = tf.transpose(siam_input, [0, 3, 1, 2]) # to [n,c,h,w]
        siam_out = siam_model.build_inf_model(inputs=siam_input) # output [n, 256, 15, 15] or [n, 256, 31, 31]
        siam_out = tf.cond(self._in_is_templar, lambda : self.build_templar_branch(siam_out), lambda : self.build_search_branch(siam_out))

        # build op to get templar kernels, this op will only be executed once for each video during init
        templar_kernels = self.get_templar_kernels(siam_out) # [15, 15, 256, n]
        self._templar_kernels = tf.get_variable(name='templar_kernels', trainable=False,
                                                initializer=tf.zeros_initializer(),
                                                shape=[15,15,256,self._num_templars]) # [15, 15, 256, n]
        self._assign_templar_kernels = tf.assign(self._templar_kernels, templar_kernels) # [15, 15, 256, n]

        # build op to get search feature maps
        search_maps = self.get_search_maps(siam_out) # [n, 256, 31, 31]

        # build op to compute cc response maps, take inputs as templar kernels, search_maps
        self._response_maps = self.compute_response(self._templar_kernels, search_maps) # outputs [n, 17, 17, 1] response map
        self._response_maps = self.upsample_response(self._response_maps) # outputs [n, 17*scale, 17*scale, 1] response map
        self._sum_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print('Built tracking graph done.')

        # create cosine window to penalize large displacements
        self._window = np.dot(np.expand_dims(np.hanning(17*self._response_up), 1),
                        np.expand_dims(np.hanning(17*self._response_up), 0))
        self._window = self._window / np.sum(self._window)  # normalize window
        self._window_influence = 0.176


        # create a session
        print('Init run session ...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        saver_siamfc = tf.train.Saver(var_list=nn.get_siamfc_vars(trainable=False))
        self._sess = tf.Session(config=sess_config)
        self._sess.run(init_op)
        saver_siamfc.restore(sess=self._sess, save_path=chkp)
        print('All variables initialized.')
        print('Init run session done.')


    def compute_response(self, templar_kernels, search_maps):
        '''
        :param templar_kernels: [15, 15, 256, n]
        :param search_maps: [n, 256, 31, 31]
        :return: [n, 17, 17, 1] response maps for n templars
        '''

        score_maps = []
        for i in range(self._num_templars):
            input_feat = search_maps[i:i + 1, :, :, :]  # [1, 256, 31, 31]
            input_filter = templar_kernels[:, :, :, i:i + 1]  # [15, 15, 256, 1]
            score_map = tf.nn.conv2d(input=input_feat, filter=input_filter, strides=[1, 1, 1, 1], padding='VALID',
                                     use_cudnn_on_gpu=True, data_format='NCHW')  # [1, 1, 17, 17]
            score_maps.append(score_map)
        final_scores = tf.concat(axis=0, values=score_maps)  # [num_templars, 1, 17, 17]
        with tf.variable_scope('heads'):
            with tf.variable_scope('final_bias'):
                final_bias = nn.get_var_cpu_no_decay(name='bias', shape=1, initializer=tf.zeros_initializer(),
                                                     training=False)  # [1]
                print('Create {0}, {1}'.format(final_bias.name, [1]))
                final_scores = tf.nn.bias_add(value=final_scores, bias=final_bias,
                                              data_format='NCHW')  # [num_templars, 1, 17, 17]
                print('Cross correlation layers built.')
        final_scores = tf.transpose(final_scores, [0,2,3,1]) # [num_templars, 17, 17, 1]
        tf.summary.image(name='score_map', tensor=final_scores)
        response_maps = tf.sigmoid(final_scores) # [num_templars, 17, 17, 1], probability maps

        return response_maps

    def build_templar_branch(self, in_tensor):
        '''
        :param in_tensor: [n, c, h, w]
        :return: [n, c, h, w]
        '''

        with tf.variable_scope('heads'):
            with tf.variable_scope('temp_adjust'):
                templar_adjust = nn.conv_layer(inputs=in_tensor, filters=[1024, 256], kernel_size=1,
                                               stride=1, l2_decay=0.0002, training=False,
                                               data_format='channels_first', pad='SAME', dilate_rate=1)
                bias_temp = nn.get_var_cpu_no_decay(name='bias', shape=256, initializer=tf.zeros_initializer(),
                                                    training=False)  # [256]
                print('Create {0}, {1}'.format(bias_temp.name, [256]))
                templar_adjust = tf.nn.bias_add(value=templar_adjust, bias=bias_temp,
                                                data_format='NCHW')  # [n, 256, 15, 15]
                templar_adjust = tf.transpose(templar_adjust, [2, 3, 1, 0])  # reshape to [15, 15, 256, n]

        return templar_adjust

    def build_search_branch(self, in_tensor):
        '''
        :param in_tensor: [n, c, h, w]
        :return: [n, c, h, w]
        '''
        with tf.variable_scope('heads'):
            with tf.variable_scope('search_adjust'):
                search_adjust = nn.conv_layer(inputs=in_tensor, filters=[1024, 256], kernel_size=1,
                                              stride=1, l2_decay=0.0002, training=False,
                                              data_format='channels_first', pad='SAME', dilate_rate=1)
                bias_search = nn.get_var_cpu_no_decay(name='bias', shape=256, initializer=tf.zeros_initializer(),
                                                      training=False)  # [256]
                print('Create {0}, {1}'.format(bias_search.name, [256]))
                search_adjust = tf.nn.bias_add(value=search_adjust, bias=bias_search,
                                               data_format='NCHW')  # [n, 256, 31, 31]

        return search_adjust

    def init_tracker(self, init_img, init_bbox):
        '''
        :param init_img: [1, h, w, 3], pixel values 0-255
        :param init_bbox: [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...], int
        :return: None
        '''

        # run session to get templar feature map, and compute templar scales
        # this will set: self._scale_templars, self._templar_kernels, self._pre_locations
        print('Init tracker ...')
        for i in range(self._num_templars):
            self._pre_locations.append(init_bbox[i])
        _, scale_templars_= self._sess.run([self._assign_templar_kernels, self._scale_templars],
                                            feed_dict={self._templar_img: init_img,
                                                       self._templar_bbox: init_bbox,
                                                       self._in_is_templar: True,
                                                       self._search_img: init_img, # will not be used in this run
                                                       self._search_bbox: init_bbox}) # will
        self._scale_templars_val = scale_templars_
        print('Init tracker done! Ready to track!')


    def reset_tracker(self):
        '''
        TODO
        Clear tracker state: template features, current search window, ...
        :return: None
        '''


    def track(self, init_img, search_img, frame_id):
        '''
        :param search_img: [1, h, w, 3], pixel values 0-255, need to minus ImageNet rgb mean outside call
        :param frame_id: frame_id to display
        :return: a list of tracked bbox coordinates in search_img as [xmin, ymin, xmax, ymax], int
        '''

        # update self._pre_locations after processing each frame, the actual displacement is the displacement_response*4/scale_templar
        print('Track frame %d'%frame_id)
        # run outputs: # [n, 33, 33, 1]
        [response_maps_, sum_op_] = self._sess.run([self._response_maps, self._sum_op],
                                        feed_dict={self._search_img: search_img,
                                                   self._search_bbox: self._pre_locations,
                                                   self._in_is_templar: False,
                                                   self._templar_img: init_img, # will not be used in this run
                                                   self._templar_bbox: self._pre_locations}) # will not be used in this run
        # process each response for each templar and get tracked bbox position
        tracked_bbox = []
        for i in range(self._num_templars):
            response = np.squeeze(response_maps_[i:i+1, :, :, :]) # [17*up_scale, 17*up_scale]
            response = (1 - self._window_influence) * response + self._window_influence * self._window # [17*up_scale, 17*up_scale]
            # find maximum response
            r_max, c_max = np.unravel_index(response.argmax(),
                                            response.shape)
            # convert coord before scaling the search image
            p_coor = np.array([r_max, c_max])
            # displacement from the center
            disp_instance_final = p_coor - np.array([136, 136]) # 17 * 16 / 2
            disp_instance_final = disp_instance_final / self._response_up
            # div by object rescale factor
            disp_instance_feat = disp_instance_final / self._scale_templars_val[i]
            # mul by stride
            disp_instance_feat = disp_instance_feat * 8
            # update the bbox position
            self._pre_locations[i][0] = self._pre_locations[i][0] + disp_instance_feat[1]
            self._pre_locations[i][1] = self._pre_locations[i][1] + disp_instance_feat[0]
            self._pre_locations[i][2] = self._pre_locations[i][2] + disp_instance_feat[1]
            self._pre_locations[i][3] = self._pre_locations[i][3] + disp_instance_feat[0]
            # save the tracked box
            tracked_bbox.append(self._pre_locations[i])

        return tracked_bbox, response, sum_op_

    def get_templar_kernels(self, siam_out):
        '''
        :param siam_out: [15, 15, 256, n]
        :return: [15, 15, 256, n] as n kernels for n templars
        '''

        return siam_out

    def get_search_maps(self, siam_out):
        '''
        :param siam_out: [n, 256, 64, 64]
        :return:
        '''

        return siam_out


    def crop_searches(self, img, search_bbox):
        '''
        :param img: [1, h, w, 3] tensor float32
        :param search_bbox: [n, 4] tensor int32, each bbox is [xmin, ymin, xmax, ymax]
        :return: [n, 255, 255, 3] tensor float32
        '''

        # self._pre_locations is a list of bbox tensors, each ele is a tensor [xmin, ymin, xmax, ymax], denotes the previous target location
        # self._scale_templars is a list of scales

        # for each templar and its scale factor, we must resize the original search image to match the templar scale
        original_h = tf.cast(tf.shape(img)[1], tf.float32)
        original_w = tf.cast(tf.shape(img)[2], tf.float32)
        rescaled_searchs = []
        for i in range(self._num_templars):
            scale_f = self._scale_templars[i]
            size = [tf.cast(original_h*scale_f, tf.int32), tf.cast(original_w*scale_f, tf.int32)]
            tmp = tf.image.resize_bilinear(images=img, size=size)
            rescaled_searchs.append(tmp)

        # get the centers of all targets in rescaled search images
        target_centers = []
        for i in range(self._num_templars):
            scale_f = self._scale_templars[i]
            center_x = tf.cast((search_bbox[i][2] - search_bbox[i][0]) / 2, tf.int32) + search_bbox[i][0]
            center_y = tf.cast((search_bbox[i][3] - search_bbox[i][1]) / 2, tf.int32) + search_bbox[i][1]
            # get center in the rescaled search img
            center_x = tf.cast(tf.cast(center_x, tf.float32)*scale_f, tf.int32)
            center_y = tf.cast(tf.cast(center_y, tf.float32)*scale_f, tf.int32)
            target_centers.append([center_x, center_y])

        # crop search patch for each target, output [256, 256] patches
        search_patches = []
        for i in range(self._num_templars):
            tmp = self.crop_single_search(rescaled_searchs[i], target_centers[i][0], target_centers[i][1])
            search_patches.append(tmp)

        return tf.concat(values=search_patches, axis=0)


    def crop_single_search(self, search_img, center_x, center_y):
        '''
        :param search_img: [1, h, w, 3]
        :param center_x:
        :param center_y:
        :return: [1, 255, 255, 3] centered at (center_x, center_y)
        '''
        img_h = tf.shape(search_img)[1]
        img_w = tf.shape(search_img)[2]
        mean_rgb = tf.reduce_mean(search_img)
        x_min, x_max = center_x - 127, center_x + 127 # may out of boundary
        y_min, y_max = center_y - 127, center_y + 127 # may out of boundary

        [new_x_min, pad_w_begin] = tf.cond(x_min < 0, lambda: self.return_zero_pad(x_min), lambda: self.return_iden_no_pad(x_min))
        [new_x_max, pad_w_end] = tf.cond(x_max >= img_w, lambda: self.return_maxW_pad(x_max, img_w),
                                         lambda: self.return_iden_no_pad(x_max))
        [new_y_min, pad_h_begin] = tf.cond(y_min < 0, lambda: self.return_zero_pad(y_min), lambda: self.return_iden_no_pad(y_min))
        [new_y_max, pad_h_end] = tf.cond(y_max >= img_h, lambda: self.return_maxH_pad(y_max, img_h),
                                         lambda: self.return_iden_no_pad(y_max))
        # do paddings, only effective if out of boundary
        search_img = search_img - mean_rgb
        search_img = tf.pad(tensor=search_img,
                            paddings=[[0,0], [pad_h_begin, pad_h_end + 10], [pad_w_begin, pad_w_end + 10], [0, 0]],
                            mode='CONSTANT', name=None, constant_values=0)
        search_img = search_img + mean_rgb
        # crop
        search_final = tf.image.crop_to_bounding_box(image=search_img, offset_height=new_y_min, offset_width=new_x_min,
                                                     target_height=255, target_width=255)

        return search_final


    def crop_templars(self, img, bbox):
        '''
        :param img: [1, h, w, 3] tensor, float32
        :param bbox: [n, 4] tensor int32, each bbox is [xmin, ymin, xmax, ymax]
        :return: [n, 127, 127, 3] tensor, [tmp0_s, tmp1_s, ..., tmpN_s] list
        '''

        mean_rgb = tf.reduce_mean(img)
        img_size = tf.shape(img)
        num_bbox = self._num_templars
        # get context margins for all bbox, and add context to them
        new_bboxs = []
        for i in range(num_bbox):
            h = bbox[i][3] - bbox[i][1]
            w = bbox[i][2] - bbox[i][0]
            context = tf.cast((h+w)/4, tf.int32)
            new_bbox = []
            new_bbox.append(bbox[i][0] - context)
            new_bbox.append(bbox[i][1] - context)
            new_bbox.append(bbox[i][2] + context)
            new_bbox.append(bbox[i][3] + context)
            new_bboxs.append(new_bbox)
        # for all non-rectangle box, convert to rectangle
        for i in range(num_bbox):
            new_bboxs[i] = self.convert_to_rectangle(new_bboxs[i])
        # now bbox may out of image boundary, pad image with enough values
        left_pad = tf.abs(tf.minimum(tf.reduce_min(new_bboxs, 0)[0] - 1, 0))
        right_pad = tf.maximum(tf.reduce_max(new_bboxs, 0)[2] - img_size[2] + 1, 0)
        top_pad = tf.abs(tf.minimum(tf.reduce_min(new_bboxs, 0)[1] - 1, 0))
        bottom_pad = tf.maximum(tf.reduce_max(new_bboxs, 0)[3] - img_size[1] + 1, 0)
        img_minus_mean = img - mean_rgb
        img_padded = tf.pad(tensor=img_minus_mean, paddings=[[0,0], [top_pad, bottom_pad],[left_pad, right_pad],[0,0]],
                          mode='CONSTANT', name=None, constant_values=0)
        img_padded = img_padded + mean_rgb
        # now that img is padded, we must update bbox positions
        for i in range(num_bbox):
            new_bboxs[i][0] = new_bboxs[i][0] + left_pad
            new_bboxs[i][1] = new_bboxs[i][1] + top_pad
            new_bboxs[i][2] = new_bboxs[i][2] + left_pad
            new_bboxs[i][3] = new_bboxs[i][3] + top_pad
        # crop templars for all bbox
        templars = []
        scale_f = []
        for i in range(num_bbox):
            height = new_bboxs[i][3]-new_bboxs[i][1]
            width = new_bboxs[i][2]-new_bboxs[i][0]
            with tf.control_dependencies([tf.debugging.assert_equal(height, width)]):
                img_padded = tf.identity(img_padded)
            tmp = tf.image.crop_to_bounding_box(image=img_padded, offset_height=new_bboxs[i][1], offset_width=new_bboxs[i][0],
                                                target_height=height,
                                                target_width=width)
            # resize
            tmp = tf.image.resize_bilinear(images=tmp, size=[127, 127])
            templars.append(tmp)
            scale_f.append(127.0/tf.cast(height, tf.float32))

        return tf.concat(values=templars, axis=0), scale_f

    def upsample_response(self, in_tensor):
        '''
        :param in_tensor: [n, s, s, 1] where s = 33
        :return: upsampled map
        '''

        new_size = [tf.shape(in_tensor)[1] * self._response_up, tf.shape(in_tensor)[1] * self._response_up]
        response_up = tf.image.resize_bicubic(images=in_tensor, size=new_size, align_corners=True, name='upsample_response')

        return response_up


    def convert_to_rectangle(self, bbox):
        '''
        :param bbox: [xmin, ymin, xmax, ymax]
        :return: rectangle box
        '''
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]

        argmin_dim = tf.math.argmin([bbox_w, bbox_h], axis=0)  # 0: shorter in width, 1: shorter in height
        extend_w_cond = tf.equal(argmin_dim, 0)  # true if extend in width dim, otherwise extend in height dim
        extend_side_cond = tf.equal(tf.math.abs(bbox_w - bbox_h) % 2, 0)  # if true, extend evenly on both side
        extend_val_left = tf.cond(extend_side_cond,
                                  lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32),
                                  lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32) + 1)
        extend_val_right = tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32)
        # get a rect bbox by extending the shorter side
        bbox_new = tf.cond(extend_w_cond,
                                   lambda: self.extend_bbox_w(bbox, extend_val_left, extend_val_right),
                                   lambda: self.extend_bbox_h(bbox, extend_val_left, extend_val_right))

        return bbox_new

    def extend_bbox_w(self, bbox, extend_val_left, extend_val_right):
        return [bbox[0] - extend_val_left, bbox[1], bbox[2] + extend_val_right, bbox[3]]

    def extend_bbox_h(self, bbox, extend_val_left, extend_val_right):
        return [bbox[0], bbox[1] - extend_val_left, bbox[2], bbox[3] + extend_val_right]

    def return_zero_pad(self, x): return [0, tf.abs(x)]
    def return_iden_no_pad(self, x): return [x, 0]
    def return_maxW_pad(self, x, w_max): return [w_max - 1, x - (w_max - 1)]
    def return_maxH_pad(self, x, h_max): return [h_max - 1, x - (h_max - 1)]

    def mean_image_subtraction(self, image, means, num_channels):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        # We have a 1-D tensor of means; convert to 3-D.
        means = tf.expand_dims(tf.expand_dims(means, 0), 0)

        return image - means
