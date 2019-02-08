'''
    The script runs inference on imagenet validation set.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
sys.path.append('..')
from core import resnet

model_params = {'load_weight': '../data/resnet_v2_imagenet_transformed/resnet50_v2.ckpt'}
model = resnet.ResNet(model_params)
output = model.build_model(tf.placeholder(tf.float32, [1, 3, None, None]),False)
dum_op = tf.add(10,20)
with tf.control_dependencies([output]):
    dum_op = tf.identity(dum_op)

init = tf.global_variables_initializer()
saver_imgnet = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # load weights
    saver_imgnet.restore(sess, model_params['load_weight'])
    print('Successfully loaded weights from {}'.format(model_params['load_weight']))
    print('dum_op: {}'.format(sess.run(dum_op)))

