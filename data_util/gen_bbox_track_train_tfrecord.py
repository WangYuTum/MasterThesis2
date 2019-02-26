'''
The files generates ImageNet15-VID train tfrecords.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, random
import tensorflow as tf
import glob
from argparse import ArgumentParser
import multiprocessing
import time

_NUM_SHARDS = 4000
_PAIRS_PER_FILE = 1000

class ImageCoder():
    def __init__(self):
        # Create a single Session to run all image coding calls on CPU
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg,
                              feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(self._cmyk_to_rgb,
                              feed_dict={self._cmyk_data: image_data})

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_to_example(image_buffer0, image_buffer1, bbox_0, bbox_1, height, width):
    '''
    :param image_buffer0:
    :param image_buffer1:
    :param bbox_0: xmax xmin ymax ymin
    :param bbox_1: xmax xmin ymax ymin
    :param height:
    :param width:
    :return: example
    '''

    example = tf.train.Example(features=tf.train.Features(feature={
        'pair/height': int64_feature(height),
        'pair/width': int64_feature(width),
        'img0/bbox/xmin': int64_feature(bbox_0[1]),
        'img0/bbox/ymin': int64_feature(bbox_0[3]),
        'img0/bbox/xmax': int64_feature(bbox_0[0]),
        'img0/bbox/ymax': int64_feature(bbox_0[2]),
        'img1/bbox/xmin': int64_feature(bbox_1[1]),
        'img1/bbox/ymin': int64_feature(bbox_1[3]),
        'img1/bbox/xmax': int64_feature(bbox_1[0]),
        'img1/bbox/ymax': int64_feature(bbox_1[2]),
        'img0/encoded': bytes_feature(image_buffer0),
        'img1/encoded': bytes_feature(image_buffer1)})) # raw image bytes buffer

    return example


def get_pair_list(source_dir):
    '''
    :param source_dir: dir that contains all train pair files, each file has 4000 pairs
    :return: a list of file paths
    '''

    files_list = [os.path.join(source_dir, name) for name in os.listdir(source_dir)]
    if len(files_list) != _NUM_SHARDS:
        raise ValueError('Number of files {} != {}'.format(len(files_list), _NUM_SHARDS))

    return files_list

def generate_train_shard(pair_file_path, out_dir, shard_id, coder):

    # get all pairs
    with open(pair_file_path, 'r') as f:
        pairs_list = f.read().splitlines()
    if len(pairs_list) != _PAIRS_PER_FILE:
        raise ValueError('Number of pairs in list {} != {}'.format(pair_file_path, _PAIRS_PER_FILE))

    # prepare output
    full_out_path = os.path.join(out_dir, 'train_' + str(shard_id) + '.tfrecord')
    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(full_out_path, options=compression_option)

    # for each pair
    for pair_line in pairs_list:
        items = pair_line.split(' ')
        # get data
        img_path_0 = items[0]
        img_path_1 = items[1]
        bbox_0 = [int(items[2]), int(items[3]), int(items[4]), int(items[5])] # xmax xmin ymax ymin
        bbox_1 = [int(items[6]), int(items[7]), int(items[8]), int(items[9])] # xmax xmin ymax ymin
        # read image buffers
        with tf.gfile.GFile(img_path_0, 'rb') as f:
            image_buffer0 = f.read()
        with tf.gfile.GFile(img_path_1, 'rb') as f:
            image_buffer1 = f.read()
        # decode image_buffer and check size
        image0 = coder.decode_jpeg(image_buffer0)
        image1 = coder.decode_jpeg(image_buffer1)
        assert len(image0.shape) == 3
        assert len(image1.shape) == 3
        assert image0.shape[2] == 3
        assert image1.shape[2] == 3
        height0 = image0.shape[0]
        width0 = image0.shape[1]
        height1 = image1.shape[0]
        width1 = image1.shape[1]
        assert height0 == height1
        assert width0 == width1

        # write to file
        example = convert_to_example(image_buffer0, image_buffer1, bbox_0, bbox_1, height0, width0)
        writer.write(example.SerializeToString())

    writer.flush()
    writer.close()
    print('Generated {}'.format(full_out_path))


def generate_train_shards_process(files_list, out_dir, shards_per_proc, proc_id):

    # each process has its own image coder/decoder
    coder = ImageCoder()
    num_shards = shards_per_proc
    if num_shards != len(files_list):
        raise ValueError('Number of files {} != shards_per_proc {}'.format(len(files_list), shards_per_proc))
    print('Start process: {}'.format(multiprocessing.current_process()))
    for local_shard_id in range(num_shards):
        shard_id = proc_id * shards_per_proc + local_shard_id
        generate_train_shard(files_list[local_shard_id], out_dir, shard_id, coder)

    print('Process finished {}'.format(multiprocessing.current_process()))


def main(args):

    # check args
    source_dir = args.source_dir
    out_dir = args.out_dir
    num_proc = args.num_proc
    num_shards = args.num_shards

    print('Generate ImageNet15-VID Training tfrecords.')
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
    pair_files_list = get_pair_list(source_dir) # should be 4000 files by default
    shards_per_proc = int(num_shards / num_proc) # should be 4000/16=250 by default
    # do not use GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # print info
    print('Number of processes: {}'.format(num_proc))
    print('Number of total_shards: {}'.format(num_shards))
    print('Number shards per processes: {}'.format(shards_per_proc))
    time.sleep(10)

    process_pool = []
    for proc_id in range(num_proc):
        start = proc_id * shards_per_proc
        end = start + shards_per_proc
        process_pool.append(multiprocessing.Process(target=generate_train_shards_process,
                                                    args=[pair_files_list[start:end], out_dir, shards_per_proc, proc_id]))

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
                            default='/usr/stud/wangyu/PycharmProjects/MasterThesis2/tmp_data/imgnet-vid/sampled_pairs')
    arg_parser.add_argument('--out-dir', dest='out_dir', default='/storage/slurm/wangyu/imagenet15_vid/tfrecord_train')
    arg_parser.add_argument('--num-proc', dest='num_proc', type=int, default=16)
    arg_parser.add_argument('--num-shards', dest='num_shards', default=4000)

    args = arg_parser.parse_args()
    main(args)