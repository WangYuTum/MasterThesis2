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
from core.nn import get_resnet50v2_backbone_vars
from core import optimizer
from core import resnet
from data_util import bbox_track_train_pipeline
from data_util.bbox_helper import draw_bbox_templar
from data_util.bbox_helper import draw_bbox_search
import time

_NUM_TRAIN = 4000000 # number of training pairs
_TRAINING = True
_NUM_GPU = 4
_NUM_SHARDS = 4000 # number of tfrecords
_BATCH_SIZE = 128  # how many pairs per iter, p6000_4x4: 128, titanx_4: 64
_PAIRS_PER_EP = 50000*4 # ideal is 4000000/batch, but too large/long; take 50000 as fc-siam paper
_BATCH_PER_GPU = int(_BATCH_SIZE / _NUM_GPU) # how many pairs per GPU
_EPOCHS = 45
_WARMUP_EP = 5 # number of epochs for warm up
_BN_MOMENTUM = 0.995 # can be 0.9 for training on large dataset, default=0.997
_BN_EPSILON = 1e-5 # default 1e-5

_OPTIMIZER = 'momentum' # can be one of the following: 'adam', 'momentum'
if _OPTIMIZER == 'adam':
    _INIT_LR = 2e-3
elif _OPTIMIZER == 'momentum':
    _INIT_LR = 5e-3
else:
    _INIT_LR = 0.01

_ADAM_EPSILON = 0.01 # try 1.0, 0.1, 0.01
_MOMENTUM_OPT = 0.9 # momentum for optimizer
_DATA_SOURCE =  '/storage/slurm/wangyu/imagenet15_vid/tfrecord_train'
_SAVE_CHECKPOINT = '/storage/slurm/wangyu/imagenet15_vid/chkp/imgnetvid_4gpu_sgd5/imgnetvid_4gpu.ckpt'  # '/work/wangyu/imgnet-vid/chkp/imgnetvid_4gpu_sgd/imgnetvid_4gpu.ckpt' #
_SAVE_SUM = '/storage/slurm/wangyu/imagenet15_vid/tfboard/imgnetvid_train_4gpu_sgd5'  # '/work/wangyu/imgnet-vid/tfboard/'
_SAVE_CHECKPOINT_EP = 1
_SAVE_SUM_ITER = 20
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

# determine number of iterations
iters_per_epoch = int(_PAIRS_PER_EP / _BATCH_SIZE) # 6250 iters/ep for batch=64, 3125 if batch=128
iters_warmup= _WARMUP_EP * iters_per_epoch
iters_total = (_EPOCHS + _WARMUP_EP) * iters_per_epoch


with tf.Graph().as_default(), tf.device('/cpu:0'):
    #######################################################################
    # Prepare data pipeline for multiple GPUs
    #######################################################################
    datasets = bbox_track_train_pipeline.build_dataset(num_gpu=_NUM_GPU,
                                                       batch_size=_BATCH_PER_GPU, # how many pairs per GPU
                                                       train_record_dir=_DATA_SOURCE,
                                                       is_training=_TRAINING, data_format='channels_first')
    iterator_gpus = [] # data iterators for different GPUs
    next_element_gpus = [] # element getter for different GPUs
    for gpu_id in range(_NUM_GPU):
        iterator_gpus.append(datasets[gpu_id].make_one_shot_iterator())
        next_element_gpus.append(iterator_gpus[gpu_id].get_next())
    # global step (shared across all GPUs), incremented automatically by 1 after each apply_gradients
    global_step = tf.get_variable(name='global_step', dtype=tf.int64, shape=[],
                                  initializer=tf.zeros_initializer(), trainable=False)
    # define optimizer (shared across all GPUs)
    if _OPTIMIZER == 'adam':
        opt = optimizer.get_adam_opt(init_lr=_INIT_LR, epsilon=_ADAM_EPSILON)
        lr = _INIT_LR
    elif _OPTIMIZER == 'momentum':
        opt, lr = optimizer.get_momentum_opt(base_lr=_INIT_LR, batches_per_epoch=iters_per_epoch, global_step=global_step,
                                             batch_size=_BATCH_SIZE, momentum=_MOMENTUM_OPT, bnorm=1)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=_INIT_LR, epsilon=_ADAM_EPSILON)
        lr = _INIT_LR

    # gradients/losses for all towers
    tower_grad = []
    tower_loss = []

    # common model attributes for all GPUs
    model_params = {'load_weight': '/storage/remote/atbeetz21/wangyu/imagenet/resnet_v2_imagenet_transformed/resnet50_v2.ckpt',
                    'batch': _BATCH_PER_GPU, # each batch actually has batch*2 images
                    'bn_momentum': _BN_MOMENTUM,
                    'bn_epsilon': _BN_EPSILON,
                    'l2_weight': 0.0002}

    #######################################################################
    # Build model on multiple GPUs
    #######################################################################
    with tf.variable_scope(tf.get_variable_scope()):  # define empty var_scope for the purpose of reusing vars on multi-gpu
        for gpu_id in range(_NUM_GPU):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('%s_%d' % ('tower', gpu_id)) as scope:  # operation scope for each gpu
                    # build model
                    print('Building model on GPU {}'.format(gpu_id))
                    model = resnet.ResNetSiam(model_params)
                    z_input = next_element_gpus[gpu_id]['templar'] # [batch, 3, 127, 127]
                    x_input = next_element_gpus[gpu_id]['search'] # [batch, 3, 255, 255]
                    reuse = False if gpu_id == 0 else True # reuse is false if first build the backbone
                    z_feat = model.build_templar(input_z=z_input, training=_TRAINING, reuse=reuse)
                    # reuse is always True
                    x_feat = model.build_search(input_x=x_input, training=_TRAINING, reuse=True)
                    score_logits = model.build_CC(z_feat=z_feat, x_feat=x_feat, training=_TRAINING) # [batch, 1, 17, 17]
                    ################# image summaries #################
                    templar_sum = draw_bbox_templar(templars=next_element_gpus[gpu_id]['templar'],
                                                    bbox_templars=next_element_gpus[gpu_id]['tight_temp_bbox'],
                                                    batch=_BATCH_PER_GPU)
                    tf.summary.image(name='templar',tensor=templar_sum)
                    search_sum = draw_bbox_search(searchs=next_element_gpus[gpu_id]['search'],
                                                  bbox_searchs=next_element_gpus[gpu_id]['tight_search_bbox'],
                                                  batch=_BATCH_PER_GPU)
                    tf.summary.image(name='search', tensor=search_sum)
                    tf.summary.image(name='score', tensor=tf.transpose(tf.cast(next_element_gpus[gpu_id]['score'], tf.uint8) * 255, [0,2,3,1]))
                    tf.summary.image(name='logits', tensor=tf.transpose(score_logits, [0,2,3,1]))
                    # get loss
                    loss = model.loss_score(score_map=score_logits, score_gt=next_element_gpus[gpu_id]['score'],
                                            score_weight=next_element_gpus[gpu_id]['score_weight'], scope=scope)
                    tower_loss.append(tf.expand_dims(loss, 0))
                    print('Model built on tower_{}'.format(gpu_id))

                    # reuse all vars for the next tower
                    tf.get_variable_scope().reuse_variables()

                    # compute grads on the current tower, and save them
                    grads_and_vars = opt.compute_gradients(loss)
                    tower_grad.append(grads_and_vars)

    # get averaged loss across towers for tensorboard display
    tower_loss = tf.concat(axis=0, values=tower_loss)
    avg_loss = tf.reduce_mean(tower_loss, axis=0)
    tf.summary.scalar(name='avg_loss', tensor=avg_loss)

    # average grads across all towers, also the synchronization point
    grads_and_vars = optimizer.average_grads(tower_grad)
    # tracking learning rate if use SGD Momentum

    if _OPTIMIZER == 'momentum':
        tf.summary.scalar(name='learning_rate', tensor=lr)

    # select vars and bn vars
    head_vars, all_vars = optimizer.apply_lr(grads_and_vars, global_step, iters_per_epoch)
    head_ops, all_ops = optimizer.apply_bn_up(global_step, iters_per_epoch)
    # update vars and bn stats
    with tf.control_dependencies(head_ops):
        update_head = opt.apply_gradients(head_vars, global_step=global_step)
    with tf.control_dependencies(all_ops):
        update_all = opt.apply_gradients(all_vars, global_step=global_step)

    # saver, summary, init
    saver_imgnet = tf.train.Saver(var_list=get_resnet50v2_backbone_vars()) # only restore backbone weights
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)
    summary_op = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    # execute graph
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement=True
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        # init all variables
        sess.run(init)
        saver_imgnet.restore(sess=sess, save_path=model_params['load_weight'])
        print('All variables initialized.')

        # get summary writer
        sum_writer = tf.summary.FileWriter(logdir=_SAVE_SUM, graph=sess.graph)

        # print info
        print('Data source: {}'.format(_DATA_SOURCE))
        print('Number of GPUs: {}'.format(_NUM_GPU))
        print('Checkpoint save per {} epoch, dir: {}'.format(_SAVE_CHECKPOINT_EP, _SAVE_CHECKPOINT))
        print('Summary save per {} iters, dir: {}'.format(_SAVE_SUM_ITER, _SAVE_SUM))
        print('Number of epoch: {}'.format(_EPOCHS + _WARMUP_EP))
        print('Number of total iterations: {}'.format(iters_total))
        print('Number of iterations per epoch: {}'.format(iters_per_epoch))
        print('Batch size: {}'.format(_BATCH_SIZE))
        print('Batch size per GPU: {}'.format(_BATCH_PER_GPU))
        if _OPTIMIZER == 'adam':
            print('Optimizer: {}, base_lr: {}, rescaled_lr: {}'.format(_OPTIMIZER, _INIT_LR, lr))
        else:
            print('Optimizer: {}, base_lr: {}, rescaled_lr: {}'.format(_OPTIMIZER, _INIT_LR, lr.eval()))
        time.sleep(3)

        # start training
        for ep_i in range(_EPOCHS + _WARMUP_EP): # in total 80 ep
            print('Epoch {}'.format(ep_i))
            for iter_i in range(iters_per_epoch):
                if global_step.eval() - 1 < iters_per_epoch * 5:
                    _, loss_v = sess.run([update_head, avg_loss])
                else:
                    _, loss_v = sess.run([update_all, avg_loss])

                # print loss
                if iter_i % 50 == 0:
                    print('iter: {}, loss: {}'.format(global_step.eval()-1, loss_v)) # global_step.eval()-1, loss_v
                # write summary
                if iter_i % _SAVE_SUM_ITER == 0 or iter_i ==0:
                    summary_out = sess.run(summary_op)
                    sum_writer.add_summary(summary_out, global_step.eval()-1)
            # save checkpoint
            if (ep_i+1) % _SAVE_CHECKPOINT_EP == 0:
                saver.save(sess=sess, save_path = _SAVE_CHECKPOINT, global_step=global_step,
                           write_meta_graph=False)
                print('Saved checkpoint after {} epochs'.format(ep_i+1))
        sum_writer.flush()
        sum_writer.close()









