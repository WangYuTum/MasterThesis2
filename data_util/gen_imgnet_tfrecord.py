'''
This file takes ImageNet data dir, some flags and output dir, and generates train/val .tfrecords files
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys, os
from argparse import ArgumentParser

# default args
IMGNET_DIR = '/work/wangyu/imagenet'
TFRECORD_TRAIN_DIR = '/work/wangyu/imagenet/tfrecord_train'
TFRECORD_VAL_DIR = '/work/wangyu/imagenet/tfrecord_val'
MAX_EXAMPLE_PER_FILE = 500


def generate_tfrecords(files_list, record_writer):
    print('ss')



def main(args):

    # parse args
    gen_train = args.gen_train
    gen_val = args.gen_val
    img_dir = args.img_dir
    train_record_dir = args.train_record_dir
    val_record_dir = args.val_record_dir
    num_proc_max = args.num_max_proc

    if not (gen_train or gen_val):
        print('Not generating train/val.')
        sys.exit(0)
    if num_proc_max < 2:
        print('Need at least 2 processes.')
        sys.exit(0)
    if not os.path.exists(img_dir):
        print('ImageNet dir does not exist: {}'.format(img_dir))
        sys.exit(0)
    if not os.path.exists(train_record_dir):
        os.makedirs(train_record_dir)
    if not os.path.exists(val_record_dir):
        os.makedirs(val_record_dir)

    # get number of files
    num_train_examples = 0
    num_val_examples = 0


    # determine number of processes
    num_train_records = num_train_examples / MAX_EXAMPLE_PER_FILE
    num_val_records = num_val_examples / MAX_EXAMPLE_PER_FILE
    if num_train_examples % MAX_EXAMPLE_PER_FILE != 0:
        num_train_records += 1
    if num_val_examples % MAX_EXAMPLE_PER_FILE != 0:
        num_val_records += 1
    total_num_records = num_train_records + num_val_records
    # records_per_proc = total_num_records / num_proc_max


    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    if gen_train:
        print('Starting generating train records.')
        # generate multiple train_x.tfrecords
    if gen_val:
        print('Starting generating val records.')
        # generate multiple train_x.tfrecords

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--imgnet-dir', dest='img_dir', default=IMGNET_DIR)
    parser.add_argument('--train-record-dir', dest='train_record_dir', default=TFRECORD_TRAIN_DIR)
    parser.add_argument('--val-record-dir', dest='val_record_dir', default=TFRECORD_VAL_DIR)
    parser.add_argument('--num-proc', dest='num_max_proc', type=int, default=2)
    parser.add_argument('gen-train', dest='gen_train', type=bool, required=True)
    parser.add_argument('gen-val', dest='gen_val', type=bool, required=True)
    args = parser.parse_args()

    main(args)