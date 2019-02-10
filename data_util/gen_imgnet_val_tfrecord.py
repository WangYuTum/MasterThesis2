'''
The files generates ImageNet train/val tfrecords.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
import glob
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

def is_png(filename):
    '''
        Check if the file is actually a png format file
        File list from:
        https://github.com/cytsai/ilsvrc-cmyk-image-list
    '''
    return 'n02105855_2933.JPEG' in filename

def is_cmyk(filename):

    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                     'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                     'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                     'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                     'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                     'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                     'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                     'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                     'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                     'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                     'n07583066_647.JPEG', 'n13037406_4650.JPEG']

    return os.path.basename(filename) in blacklist

def decode_img(filename):

    img_data = Image.open(filename)
    width, height = 0, 0
    if img_data.mode != 'RGB':
        img_data = img_data.convert('RGB')
        width, height = img_data.size
    img_data = np.array(img_data, dtype=np.uint8)

    return img_data, width, height

def png_to_jpeg(filename):

    # convert to png file
    filename = filename.replace('JPEG', 'png')
    img_data = Image.open(filename)
    width, height = 0, 0
    if img_data.mode != 'RGB':
        img_data = img_data.convert('RGB')
        width, height = img_data.size
    img_data = np.array(img_data, dtype=np.uint8)

    return img_data, width, height

def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example(img_path, label, img_data, width, height):

    colorspace = 'RGB'.encode()
    channels = 3
    image_format = 'JPEG'.encode()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/colorspace': bytes_feature(colorspace),
        'image/channels': int64_feature(channels),
        'image/class/label': int64_feature(label),
        'image/format': bytes_feature(image_format),
        'image/filename': bytes_feature(os.path.basename(img_path).encode()),
        'image/encoded': bytes_feature(img_data.flatten().tostring())})) # flatten in row-major

    return example

def generate_val_shard(val_list, out_dir, examples_per_shard, shard_id):

    img_list = val_list[0]
    label_list = val_list[1]
    num_examples = len(img_list)
    if num_examples != examples_per_shard:
        print('Error! Number of examples {} not equal to examples_per_shard {} for shard_id {}'.format(
            num_examples, examples_per_shard, shard_id))
    full_out_path = os.path.join(out_dir, 'val_'+str(shard_id)+'.tfrecord')
    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(full_out_path, options=compression_option)
    for ex_id in range(num_examples):
        label = int(label_list[ex_id])
        img_path = img_list[ex_id]
        if is_png(img_path):
            img_data, width, height = png_to_jpeg(img_path)
        else:
            img_data, width, height = decode_img(img_path) # decode to RGB, no matter it's RGB, cmyk or grayscale
        if img_path.shape[2] != 3:
            print('Image channels != 3! for {}'.format(img_path))
            sys.exit(0)
        example = convert_to_example(img_path, label, img_data, width, height)
        writer.write(example.SerializeToString())

    writer.flush()
    writer.close()
    print('Generated {}'.format(full_out_path))


def generate_val_shards_process(val_list,  out_dir, shards_per_proc, examples_per_shard, proc_id, last_proc, remain_shards):

    num_shards = shards_per_proc
    if last_proc:
        num_shards += remain_shards
    print('Start process: {}'.format(multiprocessing.current_process()))
    for local_shard_id in range(num_shards):
        shard_id = proc_id * shards_per_proc + local_shard_id
        start = local_shard_id * examples_per_shard
        if local_shard_id == num_shards - 1:
            end = len(val_list[0])
        else:
            end = start + examples_per_shard
        generate_val_shard([val_list[0][start:end], val_list[1][start:end]], out_dir, examples_per_shard, shard_id)


def main(args):

    # check args
    img_dir = args.img_dir
    out_val_dir = args.out_val_dir
    num_proc = args.num_proc
    examples_per_shard = args.examples_per_shard

    print('Generate ImageNet Validation tfrecords.')
    if not os.path.exists(out_val_dir):
        os.makedirs(out_val_dir)
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

    process_pool = []
    last_proc = False
    for proc_id in range(num_proc):
        start = proc_id * shards_per_proc * examples_per_shard
        if proc_id == num_proc - 1:
            last_proc = True
            end = num_val_imgs
        else:
            end = start + shards_per_proc * examples_per_shard
        process_pool.append(multiprocessing.Process(target=generate_val_shards_process,
                                                    args=[[val_list[0][start:end], val_list[1][start:end]],
                                                          out_val_dir, shards_per_proc, examples_per_shard, proc_id,
                                                          last_proc, remain_shards]))
    for i in range(num_proc):
        process_pool[i].start()
    for i in range(num_proc):
        process_pool[i].join()

    print('Generate val tfrecords done.')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--imgnet_dir', dest='img_dir', default='/work/wangyu/imagenet')
    arg_parser.add_argument('--out-val-dir', dest='out_val_dir', default='/work/wangyu/imagenet/tfrecord_val')
    arg_parser.add_argument('--num-proc', dest='num_proc', type=int, default=2)
    arg_parser.add_argument('--num-examples-per-shard', dest='examples_per_shard', default=400)

    args = arg_parser.parse_args()
    main(args)
