'''
    The script runs training on imagenet train set on single GPU.
    Note that the script only support training with image format channels_first.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
sys.path.append('..')
from core import optimizer
from core import resnet
from data_util import imgnet_train_pipeline
import time

_NUM_TRAIN = 1281167
_TRAINING = True
_NUM_GPU = 4
_NUM_SHARDS = 1024
_BATCH_SIZE = 256
_BATCH_PER_GPU = _BATCH_SIZE / _NUM_GPU
_EPOCHS = 100
_BN_MOMENTUM = 0.95 # can be 0.9 for training on large dataset, default=0.997
_BN_EPSILON = 1e-5
_BNORM = 512 # fixed

_OPTIMIZER = 'adam' # can be one of the following: 'adam', 'momentum'
if _OPTIMIZER == 'adam':
    _INIT_LR = 0.05 # can try 0.1 (b=128), 0.2 (b=256)
elif _OPTIMIZER == 'momentum':
    _INIT_LR = 0.256 # will be scaled to 0.064 (b=128, sgd), 0.128 (b=256, sgd), 0.256 (b=512, sgd)
else:
    _INIT_LR = 0.1

_ADAM_EPSILON = 0.01 # try 1.0, 0.1, 0.01
_MOMENTUM_OPT = 0.9 # momentum for optimizer
_DATA_SOURCE = '/storage/slurm/wangyu/imagenet/tfrecord_train'
_SAVE_CHECKPOINT = '/storage/remote/atbeetz21/wangyu/imagenet/resnet_imgnet_1gpu_scratch/imgnet_1gpu_scratch.ckpt'
_SAVE_SUM = '/storage/remote/atbeetz21/wangyu/imagenet/tfboard/imgnet_train_single_gpu'
_SAVE_CHECKPOINT_EP = 10
_SAVE_SUM_ITER = 50
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

# determine number of iterations
iters_per_epoch = int(_NUM_TRAIN / _BATCH_SIZE) + 1
iters_total = _EPOCHS * iters_per_epoch


with tf.Graph().as_default(), tf.device('/cpu:0'):
    #######################################################################
    # Prepare data pipeline for multiple GPUs
    #######################################################################
    # TODO: setup data pipeline for multi GPUs
    [dataset] = imgnet_train_pipeline.build_dataset(num_gpu=_NUM_GPU, batch_size=_BATCH_PER_GPU,
                                                    train_record_dir=_DATA_SOURCE,
                                                    is_training=_TRAINING, data_format='channels_first')
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    # global step, incremented automatically by 1 after each apply_gradients
    global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                                  initializer=tf.zeros_initializer(), trainable=False)

    # define optimizer
    if _OPTIMIZER == 'adam':
        opt = optimizer.get_adam_opt(init_lr=_INIT_LR, epsilon=_ADAM_EPSILON)
    elif _OPTIMIZER == 'momentum':
        opt = optimizer.get_momentum_opt(base_lr=_INIT_LR, batches_per_epoch=iters_per_epoch, global_step=global_step,
                                         batch_size=_BATCH_SIZE, momentum=_MOMENTUM_OPT, bnorm=_BNORM)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=_INIT_LR, epsilon=_ADAM_EPSILON)

    #######################################################################
    # Build model on single GPU
    #######################################################################
    # common attributes
    model_params = {'load_weight': '/storage/remote/atbeetz21/wangyu/imagenet/resnet_v2_imagenet_transformed/resnet50_v2.ckpt',
                    'batch': _BATCH_SIZE,
                    'init_lr': _INIT_LR,
                    'bn_momentum': _BN_MOMENTUM,
                    'bn_epsilon': _BN_EPSILON}

    with tf.device('/gpu:0'):
        print('Building model on GPU {}'.format(0))

        # build model and compute total loss
        model = resnet.ResNet(model_params)
        dense_out = model.build_model(inputs=next_element['image'], training=_TRAINING)
        total_loss = model.loss(dense_out, next_element['label'])
        # img_sum = tf.summary.image(name='img', tensor=tf.transpose(next_element['image'], [0, 2, 3, 1]), max_outputs=10)

        # compute gradient on the GPU
        grads = opt.compute_gradients(loss=total_loss)


    # apply grads to model parameters, BN moving stats dependency is handled inside the BN layer
    train_op = opt.apply_gradients(grads_and_vars=grads, global_step=global_step)

    # saver, summary, init
    saver_imgnet = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    # execute graph
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # init all variables
        sess.run(init)

        # get summary writer
        sum_writer = tf.summary.FileWriter(logdir=_SAVE_SUM, graph=sess.graph)

        # print info
        print('Data source: {}'.format(_DATA_SOURCE))
        print('Number of GPUs: {}'.format(_NUM_GPU))
        print('Checkpoint save per {} epoch, dir: {}'.format(_SAVE_CHECKPOINT_EP, _SAVE_CHECKPOINT))
        print('Summary save per {} iters, dir: {}'.format(_SAVE_SUM_ITER, _SAVE_SUM))
        print('Number of epoch: {}'.format(_EPOCHS))
        print('Number of total iterations: {}'.format(iters_total))
        print('Number of iterations per epoch: {}'.format(iters_per_epoch))
        print('Batch size: {}'.format(_BATCH_SIZE))
        time.sleep(10)

        # start training
        for ep_i in range(_EPOCHS):
            print('Epoch {}'.format(ep_i))
            for iter_i in range(iters_per_epoch):
                _, loss_v = sess.run([train_op, total_loss])
                #_ = sess.run([next_element])

                # print loss
                if iter_i % 20 == 0:
                    print('iter: {}, loss: {}'.format(global_step.eval()-1, loss_v)) # global_step.eval()-1, loss_v
                # write summary
                if iter_i % _SAVE_SUM_ITER == 0 or iter_i ==0:
                    summary_out = sess.run(summary_op)
                    sum_writer.add_summary(summary_out, global_step.eval()-1)
            # save checkpoint
            if (ep_i+1) % _SAVE_CHECKPOINT_EP == 0:
                saver_imgnet.save(sess=sess, save_path = _SAVE_CHECKPOINT, global_step=global_step,
                                  write_meta_graph=False)
                print('Saved checkpoint after {} epochs'.format(ep_i+1))
        sum_writer.flush()
        sum_writer.close()









