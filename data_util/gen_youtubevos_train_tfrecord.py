'''
The files generates Youtube VOS train tfrecords.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
from argparse import ArgumentParser
import multiprocessing
import time

_NUM_PAIRS = 285849
_NUM_SHARDS = 256 # 255*1116 + 1*1269 = 284580 + 1269
_PAIRS_PER_FILE = 1116 # 1116 pairs for 255 shards
_PAIRS_LAST_FILE = 1269 # 1269 pairs for the last shard

class ImageCoder():
    def __init__(self):
        # Create a single Session to run all image coding calls on CPU
        self._sess = tf.Session()

        # Initializes function that decodes PNG data.
        self._decode_png_data = tf.placeholder(dtype=tf.string)
        self._decode_png = tf.image.decode_png(self._decode_png_data, channels=1)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

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
    file_path = os.path.join(source_dir, 'train_pairs.txt')
    with open(file_path) as f:
        head_line = f.readline()
        data = f.read().splitlines()
    if len(data) != _NUM_PAIRS:
        raise ValueError('Number of pairs {} != {}'.format(len(data), _NUM_PAIRS))

    return data

def generate_train_shard(pairs_list, out_dir, num_examples, shard_id, coder):

    if len(pairs_list) != num_examples:
        raise ValueError('Number of pairs in shard_id {} != {}'.format(shard_id, num_examples))

    # prepare output
    full_out_path = os.path.join(out_dir, 'train_' + str(shard_id) + '.tfrecord')
    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(full_out_path, options=compression_option)

    # for each pair 'img1_dir img2_dir anno1_dir anno2_dir local_object_id'
    for pair_line in pairs_list:
        items = pair_line.split(' ')
        # get data
        img_path_0 = items[0]
        img_path_1 = items[1]
        anno_path_0 = items[2]
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

        # write to file
        example = convert_to_example(image_buffer0, image_buffer1, anno_buffer0, anno_buffer1, object_id, height0, width0)
        writer.write(example.SerializeToString())

    writer.flush()
    writer.close()
    print('Generated {}'.format(full_out_path))


def generate_train_shards_process(pairs_list, out_dir, shards_per_proc, _PAIRS_PER_FILE, _PAIRS_LAST_FILE, is_last_proc, proc_id):

    # each process has its own image coder/decoder
    coder = ImageCoder()
    num_shards = shards_per_proc
    print('Start process: {}'.format(multiprocessing.current_process()))
    for local_shard_id in range(num_shards):
        shard_id = proc_id * shards_per_proc + local_shard_id
        start = local_shard_id * _PAIRS_PER_FILE
        if local_shard_id == num_shards - 1: # the last shard within the process
            end = len(pairs_list)
            if is_last_proc: # the last process, and the last shard
                generate_train_shard(pairs_list[start:end], out_dir, _PAIRS_LAST_FILE, shard_id, coder)
            else: # not the last process, but the last shard
                generate_train_shard(pairs_list[start:end], out_dir, _PAIRS_PER_FILE, shard_id, coder)
        else: # not the last shard
            end = start + _PAIRS_PER_FILE
            generate_train_shard(pairs_list[start:end], out_dir, _PAIRS_PER_FILE, shard_id, coder)

def main(args):
    # check args
    source_dir = args.source_dir
    out_dir = args.out_dir
    num_proc = args.num_proc
    num_shards = args.num_shards

    print('Generate Youtube-VOS Training tfrecords.')
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
    pairs_list = get_pairs_list(source_dir)  # should be 285849 pairs
    shards_per_proc = int(num_shards / num_proc)  # should be 256/4=64 by default
    # do not use GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
                            default='/storage/slurm/wangyu/youtube_vos/train/')
    arg_parser.add_argument('--out-dir', dest='out_dir', default='/storage/slurm/wangyu/youtube_vos/tfrecord_train')
    arg_parser.add_argument('--num-proc', dest='num_proc', type=int, default=4)
    arg_parser.add_argument('--num-shards', dest='num_shards', default=_NUM_SHARDS)

    args = arg_parser.parse_args()
    main(args)