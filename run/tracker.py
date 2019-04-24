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
        self._mask_threshold = 0.65
        self._R_MEAN = 123.68
        self._G_MEAN = 116.78
        self._B_MEAN = 103.94
        self._CHANNEL_MEANS = [self._R_MEAN, self._G_MEAN, self._B_MEAN]
        self._response_up = 16
        self._num_templars = num_templars
        self._pre_locations = [] # np array, # store the previous bbox positions as a list, each ele is a list: [xmin, ymin, xmax, ymax]
        self._scale_templars = [] # store the scale factor for each templar as tensor
        self._scale_templars_val = None # save values of templar scale as np array
        self._templar_img = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 4], name='templars_in')
        self._templar_bbox = tf.placeholder(dtype=tf.int32, shape=[self._num_templars, 4], name='templars_bbox_in') # for N box, [xmin, ymin, xmax, ymax]
        self._in_is_templar = tf.placeholder(dtype=tf.bool, shape=[], name='input_bool')  # true if input is templar batches
        self._search_img = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='search_in')
        self._search_bbox = tf.placeholder(dtype=tf.int32, shape=[self._num_templars, 4], name='search_bbox_in') # for N box, [xmin, ymin, xmax, ymax]

        # crop templar image to multiple patches according to given bbox, and rescale to [127, 127]
        templar_imgs, self._scale_templars = self.crop_templars(self._templar_img,
                                                                self._templar_bbox)  # in [1, h, w, 4], out [n, 127, 127, 4]
        tf.summary.image(name='templar_patch', tensor=self.blend_rgb_mask(templar_imgs))
        # crop search image to multiple patches according to previous bbox predictions
        search_imgs, search_centers = self.crop_searches(self._search_img,
                                                         self._search_bbox)  # in [1, h, w, 3], out [n, 255, 255, 4]
        # tf.summary.image(name='search_patch', tensor=self.blend_rgb_mask(search_imgs))
        # normalize image mean
        templar_imgs = self.mean_image_subtraction(templar_imgs, self._CHANNEL_MEANS, 3)  # [n, 127, 127, 4]
        search_imgs = self.mean_image_subtraction(search_imgs, self._CHANNEL_MEANS, 3)  # [n, 255, 255, 4]

        # build a network that takes input as image(s), and outputs feature map(s)
        siam_model = resnet.ResNetSiam(params={'batch': self._num_templars, 'bn_epsilon': 1e-5})
        templar_imgs = tf.transpose(templar_imgs, [0, 3, 1, 2])  # input as NCHW
        search_imgs = tf.transpose(search_imgs, [0, 3, 1, 2])  # input as NCHW
        z_feat = siam_model.build_templar(input_z=templar_imgs, training=False,
                                          reuse=False)  # [15, 15, 256, num_templars]
        x_feat = siam_model.build_search(input_x=search_imgs, training=False, reuse=True)  # [num_templars, 256, 31, 31]

        # build op to get templar kernels, this op will only be executed once for each video during init
        self._templar_kernels = tf.get_variable(name='templar_kernels', trainable=False,
                                                initializer=tf.zeros_initializer(),
                                                shape=[15, 15, 256, self._num_templars])  # [15, 15, 256, num_templars]
        self._assign_templar_kernels = tf.assign(self._templar_kernels, z_feat)  # [15, 15, 256, num_templars]

        # build ops, outputs: [num_templars, 1, 17, 17], [num_templars, 63*63, 17, 17]
        self._response_maps, mask_logits = siam_model.build_cc_mask(z_feat=self._templar_kernels, x_feat=x_feat,
                                                                    training=False)
        self._response_maps = tf.transpose(self._response_maps, [0, 2, 3, 1])  # [num_templars, 17, 17, 1]
        tf.summary.image(name='score_map', tensor=self._response_maps)
        self._response_maps = tf.sigmoid(self._response_maps)  # [num_templars, 17, 17, 1], probability maps
        # select the mask where response achieves the maximum, and resize the selected masks to original image scale
        self._masks = self.select_masks(self._response_maps, mask_logits, search_imgs[:, 0:3, :, :], search_centers,
                                        self._search_img)  # [num_templars, 127, 127, 1] as probability map
        # self._masks = self.rescale_masks(self._masks) # [mask0, mask1, ...] of length num_templars, each mask has shape [h,w,1]
        # upsample response maps to get more accurate localisation bbox
        self._response_maps = self.upsample_response(self._response_maps) # outputs [n, 17*scale, 17*scale, 1] response map
        self._sum_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print('Built tracking graph done.')

        # create cosine window to penalize large displacements
        self._window = np.dot(np.expand_dims(np.hanning(17 * self._response_up), 1),
                              np.expand_dims(np.hanning(17 * self._response_up), 0))
        self._window = self._window.astype(np.float32) / np.sum(self._window).astype(np.float32)  # normalize window
        self._window_influence = 0.176

        # create a session
        print('Init run session ...')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        saver_siamfc = tf.train.Saver(var_list=nn.get_siammask_vars(trainable=False))
        self._sess = tf.Session(config=sess_config)
        self._sess.run(init_op)
        saver_siamfc.restore(sess=self._sess, save_path=chkp)
        print('All variables initialized.')
        print('Init run session done.')

    def init_tracker(self, init_img, init_bbox):
        '''
        :param init_img: [1, h, w, 4], pixel values 0-255
        :param init_bbox: [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...], int
        :return: None
        '''

        # run session to get templar feature map, and compute templar scales
        # this will set: self._scale_templars, self._templar_kernels, self._pre_locations
        print('Init tracker ...')
        for i in range(self._num_templars):
            self._pre_locations.append(init_bbox[i])
        _, scale_templars_= self._sess.run([self._assign_templar_kernels, self._scale_templars],
                                           feed_dict={self._templar_img: init_img,  # [1, h, w, 4]
                                                       self._templar_bbox: init_bbox,
                                                       self._in_is_templar: True,
                                                      self._search_img: init_img[:, :, :, 0:3],
                                                      # will not be used in this run
                                                      self._search_bbox: init_bbox})  # will not be used in this run
        self._scale_templars_val = scale_templars_
        print('Init tracker done! Ready to track!')


    def reset_tracker(self):
        '''
        TODO
        Clear tracker state: template features, current search window, ...
        :return: None
        '''


    def track(self, init_img, init_box, search_img, frame_id):
        '''
        :param search_img: [1, h, w, 3], pixel values 0-255
        :param frame_id: frame_id to display
        :return: a list of tracked bbox coordinates in search_img as [xmin, ymin, xmax, ymax], int
        '''

        # update self._pre_locations after processing each frame, the actual displacement is the displacement_response*8/scale_templar
        print('Track frame %d'%frame_id)
        # run outputs: # [n, 272, 272, 1], [n, 127, 127, 1]
        [response_maps_, masks_, sum_op_] = self._sess.run([self._response_maps, self._masks, self._sum_op],
                                        feed_dict={self._search_img: search_img,
                                                   self._search_bbox: self._pre_locations,
                                                   self._in_is_templar: False,
                                                   self._templar_img: init_img, # will not be used in this run
                                                   self._templar_bbox: init_box}) # will not be used in this run
        # process each response for each templar and get tracked bbox position
        tracked_bbox = []
        tracked_mask = []
        for i in range(self._num_templars):
            response = np.squeeze(response_maps_[i:i+1, :, :, :]) # [17*up_scale, 17*up_scale]
            response = (1 - self._window_influence) * response + self._window_influence * self._window
            mask = masks_[i]  # [h, w, 1], original search image size
            # find maximum response
            r_max, c_max = np.unravel_index(response.argmax(),
                                            response.shape)
            # convert coord before scaling the search image
            p_coor = np.array([r_max, c_max])
            # displacement from the center
            disp_instance_final = p_coor - np.array([135.5, 135.5])  # 17 * 16 / 2
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
            tracked_mask.append(mask)

        return tracked_bbox, tracked_mask, response, sum_op_

    def select_masks(self, response_maps, mask_logits, search_imgs, search_centers, search_img):
        '''
        :param response_maps: [num_templars, 17, 17, 1], probability maps
        :param mask_logits: [num_templars, 63*63, 17, 17]
        :param search_imgs: [num_templars, 3, 255, 255]
        :param search_centers: [num_templars, 2] a list keeping the search path centers in original image
        :param search_img: [1, h, w, 3] tf.float32, original search image
        :return: [num_templars, 127, 127, 1], tf.float32, probability before threshold
        '''

        original_w = tf.shape(search_img)[2]
        original_h = tf.shape(search_img)[1]

        mask_list = []
        for temp_i in range(self._num_templars):
            score_map = response_maps[temp_i:temp_i + 1, :, :, :]  # [1, 17, 17, 1]
            score_map = tf.squeeze(score_map, axis=[0, 3])  # [17, 17]
            score_map_flat = tf.reshape(score_map, [17 * 17])
            indices = tf.math.argmax(score_map_flat)
            max_h = tf.cast(indices / 17, tf.int32)
            max_w = tf.cast(indices % 17, tf.int32)
            mask_vec = mask_logits[temp_i:temp_i + 1, :, max_h:max_h + 1, max_w:max_w + 1]  # [1, 63*63, 1, 1]
            # mask_vec = mask_logits[temp_i:temp_i + 1, :, max_w:max_w + 1, max_h:max_h + 1]  # [1, 63*63, 1, 1]
            mask = tf.reshape(mask_vec, [1, 63, 63, 1])
            mask = tf.image.resize_bilinear(mask, [127, 127])  # [1,127,127,1]
            mask_prob = tf.sigmoid(mask)  # [1,127,127,1] as probability map
            mask_bin = tf.cast(tf.math.greater_equal(mask, 0.5), tf.uint8)
            # put mask in search_rgb image
            mask_center = [128, 128] + [max_h - 8, max_w - 8] * 8  # [idx_h, idx_w]
            x_min = mask_center[1] - 63
            y_min = mask_center[0] - 63
            top_pad = y_min
            left_pad = x_min
            bottom_pad = 255 - (y_min + 127)
            right_pad = 255 - (x_min + 127)
            mask_padded = tf.pad(tensor=mask_bin,
                                 paddings=[[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]],
                                 mode='CONSTANT', name=None, constant_values=0)  # [1,255,255,1]
            with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(mask_padded)[1], 255),
                                          tf.debugging.assert_equal(tf.shape(mask_padded)[2], 255)]):
                mask_padded = tf.identity(mask_padded)
            search_img = search_imgs[temp_i:temp_i + 1, :, :, :]  # [1, 3, 255, 255]
            search_img = tf.transpose(search_img, [0, 2, 3, 1])  # [1, 255, 255, 3]
            blended_search = self.blend_rgb_mask(tf.concat([search_img, tf.cast(mask_padded, tf.float32)], -1))
            tf.summary.image(name='search_patch', tensor=blended_search)

            ######################### Recover mask in original image ########################
            rescaled_h = tf.cast(tf.cast(original_h, tf.float32) * self._scale_templars[temp_i], tf.int32)
            rescaled_w = tf.cast(tf.cast(original_w, tf.float32) * self._scale_templars[temp_i], tf.int32)
            search_center = search_centers[temp_i]  # [center_x, center_y], tf.int32
            [top_bool, top_val] = tf.cond(search_center[1] - 127 < 0, lambda: [True, tf.abs(search_center[1] - 127)],
                                          lambda: [False, tf.abs(search_center[
                                                                     1] - 127)])  # if true, cut top with top_val, else pad top with top_val
            [left_bool, left_val] = tf.cond(search_center[0] - 127 < 0, lambda: [True, tf.abs(search_center[0] - 127)],
                                            lambda: [False, tf.abs(search_center[0] - 127)])
            [bottom_bool, bottom_val] = tf.cond(search_center[1] + 127 > rescaled_h - 1,
                                                lambda: [True, 127 - (rescaled_h - 1 - search_center[1])],
                                                lambda: [False, rescaled_h - (search_center[1] + 127) - 1])
            [right_bool, right_val] = tf.cond(search_center[0] + 127 > rescaled_w - 1,
                                              lambda: [True, 127 - (rescaled_w - 1 - search_center[0])],
                                              lambda: [False, rescaled_w - (search_center[0] + 127) - 1])
            # if any side needs to be cut, we do cut; the center must be updated after cut
            cut_top = tf.cond(top_bool, lambda: top_val, lambda: 0)
            cut_left = tf.cond(left_bool, lambda: left_val, lambda: 0)
            cut_bottom = tf.cond(bottom_bool, lambda: bottom_val, lambda: 0)
            cut_right = tf.cond(right_bool, lambda: right_val, lambda: 0)
            new_mask = tf.image.crop_to_bounding_box(image=mask_padded, offset_height=cut_top, offset_width=cut_left,
                                                     target_height=255 - cut_top - cut_bottom,
                                                     target_width=255 - cut_left - cut_right)
            # new_center_x = tf.cond(left_bool, lambda : search_center[0] - cut_left, lambda : search_center[0])
            # new_center_y = tf.cond(top_bool, lambda : search_center[1] - cut_top, lambda : search_center[1])
            # pad on all sides to rescaled search image size
            top_pad = tf.cond(top_bool, lambda: 0, lambda: top_val)
            left_pad = tf.cond(left_bool, lambda: 0, lambda: left_val)
            bottom_pad = tf.cond(bottom_bool, lambda: 0, lambda: bottom_val)
            right_pad = tf.cond(right_bool, lambda: 0, lambda: right_val)
            new_mask = tf.pad(tensor=new_mask,
                              paddings=[[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]],
                              mode='CONSTANT', name=None, constant_values=0)  # [1, h', w', 1]
            with tf.control_dependencies(
                    [tf.debugging.assert_equal(tf.shape(new_mask)[1], rescaled_h, message='height'),
                     tf.debugging.assert_equal(tf.shape(new_mask)[2], rescaled_w, message='width')]):
                new_mask = tf.identity(new_mask)  # [1, h', w', 1]
            # rescale to original search image size
            new_mask = tf.image.resize_bilinear(images=new_mask, size=[original_h, original_w],
                                                name='resize_search_mask')  # [1, h, w, 1]
            new_mask = tf.cast(tf.math.greater_equal(new_mask, 0.5), tf.uint8)
            mask_list.append(new_mask)

        masks = tf.concat(mask_list, axis=0)  # [num_templars, h, w, 1]

        return masks

    def rescale_masks(self, masks):
        '''
        :param masks: [num_templars, 127, 127, 1]
        :return: [mask0, mask1, ...] of length num_templars
        '''

        rescaled_masks = []
        for temp_i in range(self._num_templars):
            mask = masks[temp_i:temp_i + 1, :, :, :]  # [1,127,127,1]
            new_size = tf.cast(127 / self._scale_templars[temp_i], tf.int32)
            mask = tf.image.resize_bilinear(images=mask, size=[new_size, new_size])  # [1, h, w, 1]
            # threshold
            mask = tf.cast(tf.math.greater_equal(mask, self._mask_threshold), tf.uint8)
            rescaled_masks.append(tf.squeeze(mask, 0))

        return rescaled_masks


    def crop_searches(self, img, search_bbox):
        '''
        :param img: [1, h, w, 3] tensor float32
        :param search_bbox: [n, 4] tensor int32, each bbox is [xmin, ymin, xmax, ymax]
        :return: [n, 255, 255, 4] tensor float32
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
            tmp = self.crop_single_search(rescaled_searchs[i], target_centers[i][0],
                                          target_centers[i][1])  # [1, 255, 255, 4]
            search_patches.append(tmp)

        return tf.concat(values=search_patches, axis=0), target_centers


    def crop_single_search(self, search_img, center_x, center_y):
        '''
        :param search_img: [1, h, w, 3]
        :param center_x:
        :param center_y:
        :return: [1, 255, 255, 4] centered at (center_x, center_y)
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
                                                     target_height=255, target_width=255)  # [1, 255, 255, 3]
        # get 2d cosine window
        cos_window = np.dot(np.expand_dims(np.hanning(255), 1),
                            np.expand_dims(np.hanning(255), 0))
        cos_window = cos_window / np.sum(cos_window)  # normalize window, [255, 255]
        cos_window = np.expand_dims(np.expand_dims(cos_window, axis=-1), axis=0)  # [1, 255, 255, 1]
        search_final = tf.concat([search_final, cos_window], -1)  # [1, 255, 255, 4]

        return search_final


    def crop_templars(self, img, bbox):
        '''
        :param img: [1, h, w, 4] tensor, float32
        :param bbox: [n, 4] tensor int32, each bbox is [xmin, ymin, xmax, ymax]
        :return: [n, 127, 127, 4] tensor, [tmp0_s, tmp1_s, ..., tmpN_s] list
        '''

        mean_rgb = tf.reduce_mean(img[:, :, :, 0:3])  # mean over RGB channels
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
        img_minus_mean = img[:, :, :, 0:3] - mean_rgb  # [1,h,w,3]
        img_padded = tf.pad(tensor=img_minus_mean, paddings=[[0,0], [top_pad, bottom_pad],[left_pad, right_pad],[0,0]],
                            mode='CONSTANT', name=None, constant_values=0)
        img_padded = img_padded + mean_rgb  # [1,h,w,3]
        mask_padded = tf.pad(tensor=img[:, :, :, 3:4],
                             paddings=[[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]],
                             mode='CONSTANT', name=None, constant_values=0)  # [1,h,w,1]
        img_padded = tf.concat([img_padded, mask_padded], -1)  # [1,h,w,4]
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
                                                target_width=width)  # [1,h,w,4]
            # resize RGB and mask
            tmp_rgb = tf.image.resize_bilinear(images=tmp[:, :, :, 0:3], size=[127, 127])  # [1,h,w,3]
            tmp_mask = tf.image.resize_bilinear(images=tmp[:, :, :, 3:4], size=[127, 127])  # [1,h,w,1]
            tmp = tf.concat([tmp_rgb, tmp_mask], -1)  # [1,h,w,4]
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

    def mean_image_subtraction(self, images, means, num_channels):
        '''
        :param images: [n, h, w, 4] where n is the number of templars, last channel is mask
        :param means: [r_mean, g_mean, b_mean]
        :param num_channels: 3
        :return: [n, h, w, 4] where rgb channel is subtracted by means
        '''
        if images.get_shape().ndims != 4:
            raise ValueError('Input must be of size [n, height, width, C>0]')

        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        # extract RGB channels
        new_imgs = []
        for temp_i in range(self._num_templars):
            rgb_img = tf.squeeze(images[temp_i:temp_i + 1, :, :, 0:3], 0)  # [h, w, 3]
            means = tf.expand_dims(tf.expand_dims(means, 0), 0)  # [1,1,3]
            new_img = rgb_img - means  # [h, w, 3]
            # assemble the mask channel
            new_img = tf.concat([new_img, tf.squeeze(images[temp_i:temp_i + 1, :, :, 3:4], 0)], axis=-1)  # [h, w, 4]
            new_imgs.append(new_img)
        # stack all templars
        out_imgs = tf.stack(new_imgs)  # [n, h, w, 4]

        return out_imgs

    def blend_rgb_mask(self, img_mask):
        '''
        :param img_mask: [n, h, w, 4], tf.float32, n is the number of templars
        :return: [n, h, w, 3], tf.float32
        '''

        blended_list = []
        for temp_i in range(self._num_templars):
            mask = img_mask[temp_i:temp_i + 1, :, :, 3:4]  # [1,h,w,1], tf.float32
            # create redish mask
            mask_r = mask * 128
            mask_g = mask * 32
            mask_b = mask * 64
            mask_rgb = tf.concat([mask_r, mask_g, mask_b], axis=-1)  # [1,h,w,3], tf.float32
            rgb_img = img_mask[temp_i:temp_i + 1, :, :, 0:3]  # [1,h,w,3], tf.float32
            blended = rgb_img + mask_rgb  # [1,h,w,3], tf.float32
            blended_list.append(blended)
        blended = tf.concat(values=blended_list, axis=0)  # [n, h, w, 3], tf.float32

        return blended
