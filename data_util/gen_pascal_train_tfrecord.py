'''
The files generates PascalVOC12 train tfrecords.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
from argparse import ArgumentParser
import multiprocessing
import time
from random import shuffle
import numpy as np
from PIL import Image

_NUM_PAIRS = 0 # TODO valid number of pairs is:
_NUM_SHARDS = 0 #
_PAIRS_PER_FILE = 0 #
_PAIRS_LAST_FILE = 0 #

class ImageCoder():
    def __init__(self):
        # Create a Session to run image encoding/decoding
        self._config_gpu = tf.ConfigProto()
        self._config_gpu.gpu_options.allow_growth = True
        self._sess = tf.Session(config=self._config_gpu)

        # Initializes function that decodes PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=1)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

        # encode png
        self._encode_png_data = tf.placeholder(dtype=tf.uint8)
        self._encode_png = tf.image.encode_png(image=self._encode_png_data)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

    def decode_png(self, image_data):
        image = self._sess.run(self._decode_png,
                               feed_dict={self._decode_png_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 1
        return image

    def encode_png(self, image_data):
        image_buffer = self._sess.run(self._encode_png,
                               feed_dict={self._encode_png_data: image_data})

        return image_buffer


def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example(image_buffer0, image_buffer1, anno_buffer0, anno_buffer1, object_id, height, width):

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/object_id': int64_feature(object_id),
        'image0/encoded': bytes_feature(image_buffer0),
        'image1/encoded': bytes_feature(image_buffer1),
        'anno0/encoded': bytes_feature(anno_buffer0),
        'anno1/encoded': bytes_feature(anno_buffer1)}))

    return example

def get_pairs_list(source_dir):
    '''
    :param source_dir: dir that contains all train pairs file
    :return: a list of pairs
    '''
    file_path = os.path.join(source_dir, 'sampled_pairs.txt')
    with open(file_path) as f:
        head_line = f.readline()
        data = f.read().splitlines()
    if len(data) != _NUM_PAIRS:
        raise ValueError('Number of pairs {} != {}'.format(len(data), _NUM_PAIRS))
    shuffle(data)

    return data

def generate_train_shard(pairs_list, out_dir, num_examples, shard_id, coder):

    if len(pairs_list) != num_examples:
        raise ValueError('Number of pairs in shard_id {} != {}'.format(shard_id, num_examples))

    # prepare output
    full_out_path = os.path.join(out_dir, 'train_' + str(shard_id) + '.tfrecord')
    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(full_out_path, options=compression_option)

    # for each pair 'img1_dir, gt1_dir, img2_dir, gt2_dir, obj_id'
    count = 0
    for pair_line in pairs_list:
        items = pair_line.split(' ')
        # get data
        img_path_0 = items[0]
        img_path_1 = items[2]
        anno_path_0 = items[1]
        anno_path_1 = items[3]
        object_id = int(items[4])
        # read image buffers
        with tf.gfile.GFile(img_path_0, 'rb') as f:
            image_buffer0 = f.read()
        with tf.gfile.GFile(img_path_1, 'rb') as f:
            image_buffer1 = f.read()
        with tf.gfile.GFile(anno_path_0, 'rb') as f:
            anno_buffer0 = f.read()
        with tf.gfile.GFile(anno_path_1, 'rb') as f:
            anno_buffer1 = f.read()
        # decode image_buffer and check size
        image0 = coder.decode_jpeg(image_buffer0)
        image1 = coder.decode_jpeg(image_buffer1)
        anno0 = coder.decode_png(anno_buffer0)
        anno1 = coder.decode_png(anno_buffer1)
        assert len(image0.shape) == 3
        assert len(image1.shape) == 3
        assert image0.shape[2] == 3
        assert image1.shape[2] == 3
        assert anno0.shape[2] == 1
        assert anno1.shape[2] == 1
        height0 = image0.shape[0]
        width0 = image0.shape[1]
        height1 = image1.shape[0]
        width1 = image1.shape[1]
        assert height0 == height1
        assert width0 == width1
        assert height0 == anno0.shape[0]
        assert height0 == anno1.shape[0]
        assert width0 == anno0.shape[1]
        assert width0 == anno1.shape[1]

        # encode labels to make pixel values as the object ids
        anno0_obj = Image.open(anno_path_0)
        anno0_arr = np.expand_dims(np.array(anno0_obj), -1)
        if not (object_id in np.unique(anno0_arr)):
            raise ValueError('object id {} not in mask {}'.format(object_id, anno_path_0))
        encoded_anno0 = coder.encode_png(anno0_arr)
        anno1_obj = Image.open(anno_path_1)
        anno1_arr = np.expand_dims(np.array(anno1_obj), -1)
        if not (object_id in np.unique(anno1_arr)):
            raise ValueError('object id {} not in mask {}'.format(object_id, anno_path_1))
        encoded_anno1 = coder.encode_png(anno1_arr)

        # write to file
        example = convert_to_example(image_buffer0, image_buffer1, encoded_anno0, encoded_anno1, object_id, height0, width0)
        writer.write(example.SerializeToString())
        count += 1

    writer.flush()
    writer.close()
    print('Generated {}'.format(full_out_path))

    return count

def generate_train_shards_process(pairs_list, out_dir, shards_per_proc, _PAIRS_PER_FILE, _PAIRS_LAST_FILE, is_last_proc, proc_id):

    # each process has its own image coder/decoder
    coder = ImageCoder()
    num_shards = shards_per_proc
    print('Start process: {}'.format(multiprocessing.current_process()))
    count = 0
    for local_shard_id in range(num_shards):
        shard_id = proc_id * shards_per_proc + local_shard_id
        start = local_shard_id * _PAIRS_PER_FILE
        if local_shard_id == num_shards - 1: # the last shard within the process
            end = len(pairs_list)
            if is_last_proc: # the last process, and the last shard
                num_sample = generate_train_shard(pairs_list[start:end], out_dir, _PAIRS_LAST_FILE, shard_id, coder)
                count += num_sample
            else: # not the last process, but the last shard
                num_sample = generate_train_shard(pairs_list[start:end], out_dir, _PAIRS_PER_FILE, shard_id, coder)
                count += num_sample
        else: # not the last shard
            end = start + _PAIRS_PER_FILE
            num_sample = generate_train_shard(pairs_list[start:end], out_dir, _PAIRS_PER_FILE, shard_id, coder)
            count += num_sample
    print('Process done, number of valid samples: {}'.format(count))

def main(args):
    # check args
    source_dir = args.source_dir
    out_dir = args.out_dir
    num_proc = args.num_proc
    num_shards = args.num_shards

    print('Generate PascalVOC12 Training tfrecords.')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if num_proc < 4:
        print('num-proc must be at least 4.')
        sys.exit(0)
    if not os.path.exists(source_dir):
        print('{} does not exist.')
        sys.exit(0)
    if num_shards != _NUM_SHARDS:
        raise ValueError('Number of shards must be {}'.format(_NUM_SHARDS))
    pairs_list = get_pairs_list(source_dir)  # TODO: should be 285849 pairs
    shards_per_proc = int(num_shards / num_proc)  # TODO: should be 256/4=64 by default
    # use GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # print info
    print('Number of processes: {}'.format(num_proc))
    print('Number of total_shards: {}'.format(num_shards))
    print('Number shards per processes: {}'.format(shards_per_proc))
    time.sleep(10)

    process_pool = []
    is_last_proc = False
    for proc_id in range(num_proc):
        start = proc_id * shards_per_proc * _PAIRS_PER_FILE
        if proc_id != num_proc - 1:
            end = start + shards_per_proc * _PAIRS_PER_FILE
        else:
            end = _NUM_PAIRS
            is_last_proc = True
        process_pool.append(multiprocessing.Process(target=generate_train_shards_process,
                                                    args=[pairs_list[start:end], out_dir, shards_per_proc, _PAIRS_PER_FILE,
                                                          _PAIRS_LAST_FILE, is_last_proc, proc_id]))

    for i in range(num_proc):
        process_pool[i].start()
    start_t = time.time()
    for i in range(num_proc):
        process_pool[i].join()
    end_t = time.time()

    print('Generate train tfrecords done in {} seconds.'.format(end_t - start_t))



if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--source_dir', dest='source_dir',
                            default='/storage/slurm/wangyu/PascalVOC12/')
    arg_parser.add_argument('--out-dir', dest='out_dir', default='/storage/slurm/wangyu/PascalVOC12/tfrecord_train/')
    arg_parser.add_argument('--num-proc', dest='num_proc', type=int, default=4)
    arg_parser.add_argument('--num-shards', dest='num_shards', default=_NUM_SHARDS)

    args = arg_parser.parse_args()
    main(args)
