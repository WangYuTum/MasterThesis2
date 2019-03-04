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
from core import resnet

class Tracker():
    def __init__(self):

        # build graph once for all
        self._templar_img = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='templars_in')
        self._templar_bbox = tf.placeholder(dtype=tf.int32, shape=[None, 4], name='templars_bbox_in') # for N box
        self._search_imgs = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='search_in')
        self._search_window = tf.placeholder(dtype=tf.int32, shape=[None, 4], name='search_bbox_window') # for N search windows
        self._in_is_templar = tf.placeholder(dtype=tf.bool, shape=[], name='input_bool') # true if input is templar batches


        # TODO: crop templar image to multiple patches according to given bbox
        templar_imgs = self.crop_templars()

        # TODO: crop search image to multiple patches according to previous bbox prediction
        search_imgs = self.crop_searches()


        # build a network that takes input as image(s), and outputs feature map(s)
        def return_temp(): return templar_imgs
        def return_search(): return search_imgs
        siam_input = tf.cond(self._in_is_templar, return_temp, return_search)
        siam_model = resnet.ResNet(params={})
        # input is [n, 128, 128, 3] if templars, or [n, 256, 256, 3] if searchs
        siam_input = tf.transpose(siam_input, [0, 3, 1, 2]) # to [n,c,h,w]
        siam_out = siam_model.build_inf_model(inputs=siam_input) # output [n, 256, 32, 32] or [n, 256, 64, 64]

        # build op to get templar kernels, this op will only be executed once for each video during init
        templar_kernels = self.get_templar_kernels(siam_out) # [32, 32, 256, n]
        self._templar_kernels = tf.get_variable(name='templar_kernels', trainable=False)
        self._assign_templar_kernels = tf.assign(self._templar_kernels, templar_kernels) # [32, 32, 256, n]

        # build op to get search feature maps
        search_maps = self.get_search_maps(siam_out) # [n, 256, 64, 64]

        # build op to compute cc response maps, take inputs as templar kernels, search_maps
        # outputs [n, 33, 33, 1] response map
        response_maps = self.compute_response(self._templar_kernels, search_maps)



        # create a session
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        sess_config = tf.ConfigProto()
        sess_config.allow_soft_placement = True
        sess_config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=sess_config)

    def init_tracker(self, init_img, init_bbox):
        '''
        :param init_img: [1, h, w, 3], pixel values 0-255,
        :param init_bbox: [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...], int
        :return: None
        '''

        # run session to get templar feature map and keep it in the memory
        _ = self._sess.run([self._assign_templar_kernels],
                           feed_dict={self._templar_img: init_img,
                                      self._templar_bbox: init_bbox,
                                      self._in_is_templar: True})


    def reset_tracker(self):
        '''
        Clear tracker state: template features, current search window, ...
        :return: None
        '''


    def track(self, search_img):
        '''
        :param search_img: [h, w, 3], pixel values 0-255, int
        :return: tracked bbox coordinates in search_img as [xmin, ymin, xmax, ymax], int
        '''


    def get_templar_kernels(self, siam_out):
        '''
        :param siam_out: [n, 256, 32, 32]
        :return: [32, 32, 256, n] as n kernels for n templars
        '''

        return tf.transpose(siam_out, [2, 3, 1, 0])

    def get_search_maps(self, siam_out):
        '''
        :param siam_out: [n, 256, 64, 64]
        :return:
        '''

        return siam_out

    def compute_response(self, templar_kernels, search_maps):
        '''
        :param templar_kernels: [32, 32, 256, n]
        :param search_maps: [n, 64, 64, 256]
        :return:
        '''


