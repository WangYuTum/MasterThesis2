'''
The files generates ImageNet train/val tfrecords.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
import glob.glob
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import multiprocessing
import time


def get_val_list(img_dir):

    print('Reading file lists ...')
    img_lists = sorted(glob.glob(os.path.join(img_dir, 'ILSVRC2012_img_val', '*.JPEG')))
    label_file = os.path.join(img_dir, 'ILSVRC2012_validation_ground_truth.txt')

    with open(label_file) as t:
        label_list = t.read().splitlines()

    if len(label_list) != len(img_lists):
        print('Number of validation images {} and labels {} do not match!'.format(len(img_lists), len(label_list)))
    print('Got {} validation images/labels'.format(len(img_lists)))

    return [img_lists, label_list]


def main(args):

    # check args
    img_dir = args.img_dir
    out_val_dir = args.out_val_dir
    out_train_dir = args.out_train_dir
    num_proc = args.num_proc
    gen_train = args.gen_train
    gen_val = args.gen_val
    examples_per_shard = args.examples_per_shard

    if gen_val == 'False' and gen_train == 'False':
        print('gen-train and gen-val are both False.')
        sys.exit(0)
    if gen_train == 'True':
        print('Generate train tfrecords.')
        if not os.path.exists(out_train_dir):
            os.makedirs(out_train_dir)
    else:
        out_train_dir = None
    if gen_val == 'True':
        print('Generate val tfrecords.')
        if not os.path.exists(out_val_dir):
            os.makedirs(out_val_dir)
    else:
        out_val_dir = None
    if num_proc < 1:
        print('num-proc must be at least 1.')
        sys.exit(0)
    if not os.path.exists(img_dir):
        print('{} does not exist.')
        sys.exit(0)
    val_list = get_val_list(img_dir)
    num_val_imgs = len(val_list[0])
    num_val_shards = int(num_val_imgs / examples_per_shard)
    if num_val_imgs % examples_per_shard != 0:
        print('Number of val images {} not dividiable by examples_per_shard {}'.format(num_val_imgs, examples_per_shard))
        sys.exit(0)
    total_shards = num_val_shards
    shards_per_proc = int(total_shards / num_proc)
    remain_shards = total_shards % num_proc

    # print info
    print('Number of processes: {}'.format(num_proc))
    print('Number of examples per shard: {}'.format(examples_per_shard))
    print('Number of total_shards: {}'.format(total_shards))
    print('Number shards per processes: {}'.format(shards_per_proc))
    print('Additional number of shards for the last process: {}'.format(remain_shards))
    time.sleep(10)







if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--imgnet_dir', dest='img_dir', default='/work/wangyu/imagenet')
    arg_parser.add_argument('--out-val-dir', dest='out_val_dir', default='/work/wangyu/imagenet/tfrecord_val')
    arg_parser.add_argument('--out-train-dir', dest='out_train_dir', default='/work/wangyu/imagenet/tfrecord_train')
    arg_parser.add_argument('--num-proc', dest='num_proc', type=int, default=2)
    arg_parser.add_argument('--num-examples-per-shard', dest='examples_per_shard', default=400)
    arg_parser.add_argument('--gen-train', dest='gen_train', required=True)
    arg_parser.add_argument('--gen-val', dest='gen_val', required=True)

    args = arg_parser.parse_args()
    main(args)
