'''
    The script runs inference on imagenet validation set on a single GPU.
    Note that the script only support inference on single GPU with image format channels_first.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
import tensorflow as tf
sys.path.append('..')
from core import resnet
from data_util import imgnet_val_pipeline
import numpy as np

_NUM_VAL = 50000
_BATCH_SIZE = 100
_TRAINING = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True


# build data pipeline (the labels are from ImageNet2012 official dataset)
if _NUM_VAL % _BATCH_SIZE != 0:
    raise ValueError('Number of validation images {} not dividable by batch size {}'.format(_NUM_VAL, _BATCH_SIZE))
dataset = imgnet_val_pipeline.build_dataset(val_record_dir='/work/wangyu/imagenet/tfrecord_val',
                                            batch=_BATCH_SIZE, is_training=_TRAINING,
                                            data_format='channels_first')
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


# build resnet model
model_params = {'load_weight': '../data/resnet_v2_imagenet_transformed/resnet50_v2.ckpt',
                'batch': _BATCH_SIZE}
model = resnet.ResNet(model_params)
dense_out = model.build_model(inputs=next_element['image'], training=_TRAINING)
# img_sum = tf.summary.image(name='val_img', tensor=tf.transpose(next_element['image'], [0, 2, 3, 1]), max_outputs=10)
pred_label = model.inference(dense_out) # return a vector [batch]
saver_imgnet = tf.train.Saver()
# merged_sum = tf.summary.merge_all()
init = tf.global_variables_initializer()


with tf.Session(config=config_gpu) as sess:
    # sum_writer = tf.summary.FileWriter(logdir='../data/tfboard/imgnet_val', graph=sess.graph)
    sess.run(init)

    # load weights
    saver_imgnet.restore(sess, model_params['load_weight'])
    print('Successfully loaded weights from {}'.format(model_params['load_weight']))

    # do inference
    print('Start inference ..., batch size = {}'.format(_BATCH_SIZE))
    num_runs = int(_NUM_VAL / _BATCH_SIZE) # default 500 iterations
    num_correct = 0
    for run_i in range(num_runs):
        print('Iteration {}'.format(run_i))
        pred_label_, next_element_ = sess.run([pred_label, next_element])
        # sum_writer.add_summary(merged_sum_, 0)
        # convert to numpy array
        pred_label_ = np.array(pred_label_, dtype=np.int32).reshape(-1) # to a np vector
        gt_label_ = np.array(next_element_['label'], dtype=np.int32).reshape(-1)
        # get num correct predictions
        num_correct += np.sum(pred_label_ == gt_label_)
    print('Inference done. Accuracy: {}'.format(num_correct / _NUM_VAL))



