'''
The files generates ImageNet train tfrecords.
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

def get_train_list(img_dir):

    print('Reading file lists ...')
    # img_dir = '/usr/data/cvpr_shared/common_datasets/imagenet/Data' + '/train/n03062245/n03062245_4620.JPEG'
    img_lists = glob.glob(os.path.join(img_dir, 'train', '*', '*.JPEG'))
    synsets_list = [os.path.basename(os.path.dirname(f)) for f in img_lists]
    training_shuffle_idx = make_shuffle_idx(len(img_lists)) # shuffle the images

    # get shuffle image lists and synset lists
    training_files = [img_lists[i] for i in training_shuffle_idx]
    training_synsets = [synsets_list[i] for i in training_shuffle_idx]

    # create an unique label for each unique synset_label
    labels_dict = {v: k + 1 for k, v in enumerate(sorted(set(training_synsets)))}
    print('Number of labels: {}'.format(len(labels_dict)))
    # create label lists for the training files
    training_labels = [labels_dict[training_synsets[i]] for i in range(len(training_synsets))]

    # check
    if len(training_files) != len(training_labels):
        raise ValueError('Number of images {} != number of labels {}'.format(len(training_files), len(training_labels)))
    print('Got {} training images/labels'.format(len(training_files)))

    return [training_files, training_labels]


def make_shuffle_idx(n):
    order = list(range(n))
    random.shuffle(order)
    return order

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
        'image/encoded': bytes_feature(img_data)})) # raw image bytes buffer

    return example

def generate_train_shard(train_list, out_dir, exp_examples, shard_id, coder):

    img_list = train_list[0]
    label_list = train_list[1]
    num_examples = len(img_list)
    if num_examples != exp_examples:
        print('Error! Number of examples {} not equal to exp_examples {} for shard_id {}'.format(
            num_examples, exp_examples, shard_id))
    full_out_path = os.path.join(out_dir, 'train_'+str(shard_id)+'.tfrecord')
    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    writer = tf.python_io.TFRecordWriter(full_out_path, options=compression_option)
    for ex_id in range(num_examples):
        label = int(label_list[ex_id])
        img_path = img_list[ex_id]
        with tf.gfile.GFile(img_path, 'rb') as f:
            image_buffer = f.read() # it's okay if the raw image is grayscale since we will decode it as rgb during train/inf
        if is_png(img_path):
            image_buffer = coder.png_to_jpeg(image_buffer)
        elif is_cmyk(img_path):
            image_buffer = coder.cmyk_to_rgb(image_buffer) # encode to rgb space

        # decode image_buffer and check size
        image = coder.decode_jpeg(image_buffer)
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        assert image.shape[2] == 3

        # write to file
        example = convert_to_example(img_path, label, image_buffer, width, height)
        writer.write(example.SerializeToString())

    writer.flush()
    writer.close()
    print('Generated {}'.format(full_out_path))


def generate_train_shards_process(train_list,  out_dir, shards_per_proc, examples_per_shard, proc_id, last_proc, remain_shards, examples_last_shard):

    # each process has its own image coder/decoder
    coder = ImageCoder()
    num_shards = shards_per_proc
    if last_proc:
        num_shards += remain_shards
    print('Start process: {}'.format(multiprocessing.current_process()))
    for local_shard_id in range(num_shards):
        shard_id = proc_id * shards_per_proc + local_shard_id
        start = local_shard_id * examples_per_shard
        if local_shard_id == num_shards - 1: # the last shard within the process
            end = len(train_list[0])
            if last_proc: # the last process, and the last shard
                generate_train_shard([train_list[0][start:end], train_list[1][start:end]], out_dir, examples_last_shard,
                                   shard_id, coder)
            else: # not the last process, but the last shard
                generate_train_shard([train_list[0][start:end], train_list[1][start:end]], out_dir, examples_per_shard,
                                   shard_id, coder)
        else: # not the last shard
            end = start + examples_per_shard
            generate_train_shard([train_list[0][start:end], train_list[1][start:end]], out_dir, examples_per_shard, shard_id, coder)


def main(args):

    # check args
    img_dir = args.img_dir
    out_train_dir = args.out_train_dir
    num_proc = args.num_proc
    num_shards = args.num_shards

    print('Generate ImageNet Training tfrecords.')
    if not os.path.exists(out_train_dir):
        os.makedirs(out_train_dir)
    if num_proc < 4:
        print('num-proc must be at least 4.')
        sys.exit(0)
    if not os.path.exists(img_dir):
        print('{} does not exist.')
        sys.exit(0)
    [img_list, label_list] = get_train_list(img_dir)
    num_train_imgs = len(img_list)
    examples_per_shard = int(num_train_imgs / num_shards)
    examples_last_shard = (num_train_imgs - examples_per_shard * num_shards) + examples_per_shard
    total_shards = num_shards
    shards_per_proc = int(total_shards / num_proc)
    remain_shards = total_shards % num_proc
    # do not use GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # print info
    print('Number of processes: {}'.format(num_proc))
    print('Number of total_shards: {}'.format(total_shards))
    print('Number of examples per shard: {}'.format(examples_per_shard))
    print('Number of examples for last shard: {}'.format(examples_last_shard))
    print('Number shards per processes: {}'.format(shards_per_proc))
    print('Number of shards for last process: {}'.format(remain_shards + shards_per_proc))
    time.sleep(10)

    process_pool = []
    last_proc = False
    for proc_id in range(num_proc):
        start = proc_id * shards_per_proc * examples_per_shard
        if proc_id == num_proc - 1:
            last_proc = True
            end = num_train_imgs
        else:
            end = start + shards_per_proc * examples_per_shard
        process_pool.append(multiprocessing.Process(target=generate_train_shards_process,
                                                    args=[[img_list[start:end], label_list[start:end]],
                                                          out_train_dir, shards_per_proc, examples_per_shard, proc_id,
                                                          last_proc, remain_shards, examples_last_shard]))

    for i in range(num_proc):
        process_pool[i].start()
    start_t = time.time()
    for i in range(num_proc):
        process_pool[i].join()
    end_t = time.time()

    print('Generate train tfrecords done in {} seconds.'.format(end_t - start_t))

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--imgnet_dir', dest='img_dir', default='/usr/data/cvpr_shared/common_datasets/imagenet/Data')
    arg_parser.add_argument('--out-val-dir', dest='out_train_dir', default='/storage/slurm/wangyu/imagenet/tfrecord_train')
    arg_parser.add_argument('--num-proc', dest='num_proc', type=int, default=16)
    arg_parser.add_argument('--num-shards', dest='num_shards', default=1024)

    args = arg_parser.parse_args()
    main(args)

