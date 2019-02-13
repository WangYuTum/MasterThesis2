'''
    The script runs inference on imagenet validation set on multiple GPUs.
    Note that the script only support inference with image format channels_first.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
sys.path.append('..')
from core import resnet
from data_util import imgnet_val_pipeline
import numpy as np
import time

_NUM_VAL = 50000
_TRAINING = False
_NUM_GPU = 2
_NUM_SHARDS = 128
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
# different number of GPUs corresponding to different [batch_size, num_runs] to prevent data pipeline exceptions
run_config_table = {1: [100, 500], 2: [150, 167], 4: [150, 84], 8: [155, 41]}

# number of shard must be dividable by number of GPUs
if _NUM_SHARDS % _NUM_GPU != 0:
    raise ValueError('Number of shards {} must be dividable by number of GPUs {}'.format(_NUM_SHARDS, _NUM_GPU))
batch_size, num_runs = run_config_table[_NUM_GPU]


# We only need a single graph
with tf.Graph().as_default(), tf.device('/cpu:0'):
    #######################################################################
    # Prepare data pipeline for multiple GPUs
    #######################################################################
    # build data pipeline for multiple GPUs
    dataset_gpus = imgnet_val_pipeline.build_dataset(num_gpu=_NUM_GPU, batch=batch_size,
                                                     val_record_dir='/storage/slurm/wangyu/imagenet/tfrecord_val',
                                                     is_training=False, data_format='channels_first')
    iterator_gpus = [] # data iterators for different GPUs
    next_element_gpus = [] # element getter for different GPUs
    for gpu_id in range(_NUM_GPU):
        iterator_gpus.append(dataset_gpus[gpu_id].make_one_shot_iterator())
        next_element_gpus.append(iterator_gpus[gpu_id].get_next())

    #######################################################################
    # Build model on multiple GPUs
    #######################################################################
    # common attributes
    model_params = {'load_weight': '/storage/remote/atbeetz21/wangyu/imagenet/resnet_v2_imagenet_transformed/resnet50_v2.ckpt',
                'batch': batch_size}
    tower_correct_list = []
    with tf.variable_scope(tf.get_variable_scope()): # define empty var_scope for the purpose of reusing vars on multi-gpu
        for gpu_id in range(_NUM_GPU):
            with tf.device('/gpu:%d'%gpu_id):
                with tf.name_scope('%s_%d'%('tower', gpu_id)) as scope: # operation scope for each gpu
                    # build model
                    print('Building model on GPU {}'.format(gpu_id))
                    model = resnet.ResNet(model_params)
                    dense_out = model.build_model(inputs=next_element_gpus[gpu_id]['image'], training=_TRAINING)
                    pred_label = model.inference(dense_out)  # return a vector [batch]
                    gt_label = tf.reshape(next_element_gpus[gpu_id]['label'], [-1])  # to a vector [batch]
                    tower_correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(pred_label, tf.int32), gt_label), tf.int32))
                    tower_correct_list.append(tower_correct)

                    # reuse var for the next tower
                    tf.get_variable_scope().reuse_variables()

    # aggregate number of correct predictions from multiple towers, also the synchronization point
    num_correct = tf.reduce_sum(tower_correct_list)

    dump_sum = tf.summary.scalar(name='dump', tensor=1.0)
    merged_sum = tf.summary.merge_all()
    saver_imgnet = tf.train.Saver()
    init = tf.global_variables_initializer()

    # print info
    print('Number of GPUs: {}'.format(_NUM_GPU))
    print('Batch size per GPU: {}'.format(batch_size))
    print('Number of iterations: {}'.format(num_runs))

    # run inference graph
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sum_writer = tf.summary.FileWriter(logdir='/storage/slurm/wangyu/imagenet/tfboard/imgnet_val', graph=sess.graph)
        sess.run(init)

        # load weights
        saver_imgnet.restore(sess, model_params['load_weight'])
        print('Successfully loaded weights from {}'.format(model_params['load_weight']))

        sum_correct = 0
        print('Start inference ...')
        # run inference until all GPUs finished examples
        start_t = time.time()
        for run_i in range(num_runs):
            print('Iter {}'.format(run_i))
            num_correct_, merged_sum_ = sess.run([num_correct, merged_sum])
            print('Num correct: {}'.format(num_correct_))
            sum_correct += num_correct_
            sum_writer.add_summary(merged_sum_, run_i)
        end_t = time.time()
        print('Inference done in {} seconds. Accuracy: {}, Num correct: {}'.format(end_t-start_t, sum_correct / _NUM_VAL, sum_correct))



