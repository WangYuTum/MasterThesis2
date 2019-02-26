'''
    The file implements cross-correlation bbox tracking train data pipeline.

    Some functions/methods are directly copied from tensorflow official resnet model:
    https://github.com/tensorflow/models/tree/r1.12.0/official/resnet

    Therefore the code must be used under Apache License, Version 2.0 (the "License"):
    http://www.apache.org/licenses/LICENSE-2.0
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

_DEFAULT_SIZE = 256
_NUM_CHANNELS = 3
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_NUM_TRAIN = 4000000 # number of training pairs, equal to number of sampled pairs
_NUM_SHARDS = 4000 # number of tfrecords in total, each tfrecord has 1000 pairs

"""
The tfrecord files must provide the following contents, each example have:
* A pair of images (templar image/bbox, search image/bbox)
* Height, width of templar/search images
* (Optional) Segmentation mask of object in templar/search images  

Pre-processing of training pairs for bbox tracking:
*********************************** Templar image ****************************************
* Get tight bbox, randomly shift +-8 pixels
* Pad image to [2500, 2500] with mean RGB values
* Crop to 256x256:
    * get tight bbox [w, h]
    * compute context margin p = (w+h)/4
    * extend bbox to [w+2p, h+2p], and get min(w+2p, h+2p)
    * extend bbox to [D, D] by adding the shorter side with max(w+2p, h+2p) - min(w+2p, h+2p)
    * crop [D, D] and rescale to [128, 128], get the rescale factor [s]
    * pad boundaries to [256,256] with mean RGB values

*********************************** Search image ****************************************
* Get tight bbox of the corresponding object in templar image
* Randomly rescale in range(s*0.8, s*1.2), and update bbox position; [s] is computed during pre-process templar image
* Pad image to [2500, 2500] with mean RGB values
* Set bbox as the center and crop the image to [256, 256] so that search target is centered in the image


*********************************** Loc GT ****************************************
* Make a zeros mask [256, 256], tf.int32
* Set central [16, 16] pixels to one
* Make balanced weight mask for the GT


The above pre-processing preserve aspect-ratio.
Images are additionally flipped randomly (templar & search flip together).
Images undergo mean color subtraction from ImageNet12.
"""

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""

  return [ os.path.join(data_dir, 'train_'+str(shard_id)+'.tfrecord') for shard_id in range(_NUM_SHARDS)]

def reformat_channel_first(example_dict):

    templar = tf.transpose(example_dict['templar'], [2, 0, 1]) # from [h, w, c] to [c, h, w]
    search = tf.transpose(example_dict['search'], [2, 0, 1])
    score = tf.transpose(example_dict['score'], [2, 0, 1])
    score_weight = tf.transpose(example_dict['score_weight'], [2, 0, 1])

    dict = {'templar':templar, 'search':search, 'score':score, 'score_weight':score_weight}

    return dict

def parse_func(example_proto):
    """
        Callable func to be fed to dataset.map()
    """

    parsed_dict = parse_record(raw_record=example_proto,
                               is_training=True,
                               dtype=tf.float32) # images/score_weight: tf.float32, [h, w, 3]; score: tf.int32

    return parsed_dict

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of templar/search image pair.
  The image buffers are passed to be pre-processed
  Args:
    raw_record: scalar Tensor tf.string containing a serialized Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.
  Returns:
    Parsed example of dict type: {'templar':templar, 'search':search, 'score':score, 'score_weight':score_weight}
  """

  templar_buffer, search_buffer, templar_bbox, search_bbox = parse_example_proto(raw_record)
  templar_img, search_img, score, score_weight = preprocess_pair(templar_buffer=templar_buffer,
                                                                 search_buffer=search_buffer,
                                                                 templar_bbox=templar_bbox,
                                                                 search_bbox=search_bbox,
                                                                 num_channels=_NUM_CHANNELS,
                                                                 is_training=is_training)
  templar_img = tf.cast(templar_img, dtype)
  search_img = tf.cast(search_img, dtype)
  score = tf.cast(score, tf.int32)
  score_weight = tf.cast(score_weight, dtype)

  dict = {'templar': templar_img, 'search': search_img, 'score': score, 'score_weight': score_weight}

  return dict


def parse_example_proto(raw_record):
    '''
    :param raw_record: scalar Tensor tf.string containing a serialized Example protocol buffer.
    :return:
        templar_buffer: Tensor tf.string containing the contents of a JPEG file.
        search_buffer: Tensor tf.string containing the contents of a JPEG file.
        templar_bbox: Tensor tf.string containing the bbox in templar image
        search_bbox: Tensor tf.string containing the bbox in search image
    '''

    feature_map = {
        'pair/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'pair/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img0/bbox/xmin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img0/bbox/ymin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img0/bbox/xmax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img0/bbox/ymax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img1/bbox/xmin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img1/bbox/ymin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img1/bbox/xmax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img1/bbox/ymax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'img0/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'img1/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    features = tf.parse_single_example(raw_record, feature_map)
    bbox0 = [features['img0/bbox/xmin'], features['img0/bbox/ymin'],
             features['img0/bbox/xmax'], features['img0/bbox/ymax']]
    bbox1 = [features['img1/bbox/xmin'], features['img1/bbox/ymin'],
             features['img1/bbox/xmax'], features['img1/bbox/ymax']]
    img0_buffer = features['img0/encoded']
    img1_buffer = features['img1/encoded']

    # randomly select one image/bbox as templar
    def true_fn(): return img0_buffer, bbox0, img1_buffer, bbox1
    def false_fn(): return img1_buffer, bbox1, img0_buffer, bbox0
    templar_img, templar_box, search_img, search_box = tf.cond(tf.random.uniform([]) < 0.5, true_fn(), false_fn())

    return templar_img, search_img, templar_box, search_box

def mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.
  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.
  Returns:
    the centered image.
  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)

  return image - means

def image_pad(image, pad_value, out_size):
    '''
    :param image: Tensor of shape [H,W,3], tf.uint8
    :param pad_value: Scalar value, int
    :param out_size: desired output size in height/width
    :return: padded image of shape [out_size, out_size, 3]
    '''

    if image.get_shape().ndim != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    if image.dtype != tf.uint8:
        raise ValueError('Input must be of type tf.uint8')
    img_height = image.get_shape()[0]
    img_width = image.get_shape()[1]

    pad_h_total = tf.cast(out_size - img_height, tf.int32)
    pad_w_total = tf.cast(out_size - img_width, tf.int32)

    pad_h_cond = tf.equal(pad_h_total % 2, 0)
    pad_h_begin = tf.cond(pad_h_cond,
                          lambda: tf.cast(pad_h_total / 2, tf.int32),
                          lambda: tf.cast(pad_h_total / 2, tf.int32) + 1)
    pad_h_end = tf.cond(pad_h_cond,
                        tf.cast(pad_h_total / 2, tf.int32),
                        tf.cast(pad_h_total / 2, tf.int32))

    pad_w_cond = tf.equal(pad_w_total % 2, 0)
    pad_w_begin = tf.cond(pad_w_cond,
                          lambda: tf.cast(pad_w_total / 2, tf.int32),
                          lambda: tf.cast(pad_w_total / 2, tf.int32) + 1)
    pad_w_end = tf.cond(pad_w_cond,
                        lambda: tf.cast(pad_w_total / 2, tf.int32),
                        lambda: tf.cast(pad_w_total / 2, tf.int32))

    pad_value = tf.cast(pad_value, tf.uint8)
    image = tf.pad(tensor=image, paddings=[[pad_h_begin, pad_h_end],[pad_w_begin, pad_w_end],[0,0]],
                   mode='CONSTANT', name=None, constant_values=pad_value)

    return image


def distort_bounding_box(input_bbox, random_shift):
    '''
    :param input_bbox: [xmin, ymin, xmax, ymax]
    :param random_shift: integer
    :return: [xmin', ymin', xmax', ymax']
    '''

    h_rand = np.random.randint(low=-random_shift, high=random_shift+1, size=None)
    w_rand = np.random.randint(low=-random_shift, high=random_shift+1, size=None)

    return [input_bbox[0]+w_rand, input_bbox[1]+h_rand, input_bbox[2]+w_rand, input_bbox[3]+h_rand]

def extend_bbox_w(templar_bbox, extend_val_left, extend_val_right):
    '''
    :param templar_bbox: list [xmin, ymin, xmax, ymax], int
    :param extend_val_left:
    :param extend_val_right:
    :return:
    '''

    # ymin, ymax stay the same
    return [templar_bbox[0]-extend_val_left, templar_bbox[1], templar_bbox[2]+extend_val_right, templar_bbox[3]]

def extend_bbox_h(templar_bbox, extend_val_left, extend_val_right):
    '''
    :param templar_bbox: list [xmin, ymin, xmax, ymax], int
    :param extend_val_left:
    :param extend_val_right:
    :return:
    '''

    # xmin, xmax stay the same
    return [templar_bbox[0], templar_bbox[1]-extend_val_left, templar_bbox[2], templar_bbox[3]+extend_val_right]

def preprocess_pair(templar_buffer, search_buffer, templar_bbox, search_bbox, num_channels, is_training=True):
  """Preprocesses the give templar/search image buffers and the corresponding bbox.
  Args:
    templar_buffer: scalar string Tensor representing the raw JPEG image buffer.
    search_buffer: scalar string Tensor representing the raw JPEG image buffer.
    templar_bbox: list [xmin, ymin, xmax, ymax], int
    search_bbox: list [xmin, ymin, xmax, ymax], int
    num_channels: Integer depth of the image buffer for decoding.
    is_training: `True` if we're pre-processing the image for training and `False` otherwise.

  Returns:
    Pre-processed:
        templar_img: [256, 256, 3]
        search_img: [256, 256, 3]
        score: [256, 256, 1]
        score_weight: [256, 256, 1]
  """

  '''
  *********************************** Templar image ****************************************
  * Get tight bbox, randomly shift +-8 pixels
  * Pad image to [2500, 2500] with mean RGB values
  * Crop to 256x256:
      * get tight bbox [w, h]
      * compute context margin p = (w+h)/4
      * extend bbox to [w+2p, h+2p], and get min(w+2p, h+2p)
      * extend bbox to [D, D] by adding the shorter side with max(w+2p, h+2p) - min(w+2p, h+2p)
      * crop [D, D] and rescale to [128, 128], get the rescale factor [s]
      * pad boundaries to [256,256] with mean RGB values


  *********************************** Search image ****************************************
  * Get tight bbox of the corresponding object in templar image
  * Randomly rescale in range(s*0.8, s*1.2), and update bbox position; [s] is computed during pre-process templar image
  * Pad image to [2500, 2500] with mean RGB values
  * Set bbox as the center and crop the image to [256, 256] so that search target is centered in the image
  '''

  # decode image buffers
  templar_img = tf.image.decode_jpeg(templar_buffer, channels=num_channels) # uint8
  search_img = tf.image.decode_jpeg(search_buffer, channels=num_channels) # uint8

  ######################################## Process Templar #############################################
  # Get tight bbox, randomly shift +-8 pixels
  templar_bbox = distort_bounding_box(input_bbox=templar_bbox, random_shift=8) # new box [xmin, ymin, xmax, ymax]
  bbox_h = templar_bbox[3] - templar_bbox[1]
  bbox_w = templar_bbox[2] - templar_bbox[0]
  # Pad image to [2500, 2500] with mean RGB values
  mean_rgb = tf.reduce_mean(templar_img) # tf.uint8
  templar_img = image_pad(image=templar_img, pad_value=mean_rgb, out_size=2500)
  # get context margin and compute new bbox
  p = tf.cast((bbox_h + bbox_w) / 4, tf.int32)
  argmin_dim = tf.math.argmin([bbox_w, bbox_h], axis=0) # 0: shorter in width, 1: shorter in height
  extend_w_cond = tf.equal(argmin_dim, 0) # true if extend in width dim, otherwise extend in height dim
  extend_side_cond = tf.equal(tf.math.abs(bbox_w-bbox_h) % 2, 0) # if true, extend evenly on both side
  extend_val_left = tf.cond(extend_side_cond,
                            lambda: tf.cast(tf.math.abs(bbox_w-bbox_h) / 2, tf.int32),
                            lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32) + 1)
  extend_val_right = tf.cast(tf.math.abs(bbox_w-bbox_h) / 2, tf.int32)
  # get a rect bbox by extending the shorter side
  templar_bbox_new = tf.cond(extend_w_cond, lambda: extend_bbox_w(templar_bbox, extend_val_left, extend_val_right),
                             lambda: extend_bbox_h(templar_bbox, extend_val_left, extend_val_right))
  # add the context margin on all sides
  templar_bbox_new = [templar_bbox_new[0]-p, templar_bbox_new[1]-p, templar_bbox_new[2]+p, templar_bbox_new[3]+p]
  # crop the image
  croped_templar = tf.image.crop_to_bounding_box(image=templar_img, offset_height=templar_bbox_new[1],
                                                 offset_width=templar_bbox_new[0],
                                                 target_height=templar_bbox_new[3]-templar_bbox_new[1],
                                                 target_width=templar_bbox_new[2]-templar_bbox_new[0])
  with tf.control_dependencies(tf.debugging.assert_equal(templar_bbox_new[3] - templar_bbox_new[1],
                                                         templar_bbox_new[2] - templar_bbox_new[0])):
    # rescale to [128, 128], get the scale factor
    scale_s = 128.0 / tf.cast(templar_bbox_new[3] - templar_bbox_new[1], tf.float32)
    croped_templar = tf.image.resize_bilinear(images=tf.expand_dims(croped_templar, axis=0),
                                              size=[128, 128])
    croped_templar = tf.squeeze(croped_templar) # [h, w, 3]
  # pad boundary to [256, 256]
  templar_final = image_pad(image=croped_templar, pad_value=mean_rgb, out_size=256)
  # check size
  with tf.control_dependencies([tf.debugging.assert_equal(templar_final.get_shape()[0], 256),
                                tf.debugging.assert_equal(templar_final.get_shape()[1], 256),
                                tf.debugging.assert_equal(templar_final.get_shape()[2], 3)]):
      templar_final = tf.identity(templar_final)

  ######################################## Process Search image #############################################
  # Get rgb mean
  mean_rgb = tf.reduce_mean(search_img)  # tf.uint8
  # Get random scale factor
  rescale_factor = scale_s * float(np.random.randint(low=8, high=13, size=None)) / 10.0
  # Get rescaled bbox position, and the image
  search_bbox = tf.cast(search_bbox * rescale_factor, tf.int32)
  new_height = tf.cast(tf.cast(search_img.get_shape()[0], tf.float32) * scale_s, tf.int32)
  new_width = tf.cast(tf.cast(search_img.get_shape()[1], tf.float32) * scale_s, tf.int32)
  search_img = tf.image.resize_bilinear(images=tf.expand_dims(search_img, axis=0),
                                            size=[new_height, new_width])
  search_img = tf.squeeze(search_img)  # [h, w, 3]
  # pad to [2500, 2500]
  search_img = image_pad(image=search_img, pad_value=mean_rgb, out_size=2500)
  # Crop image around new bbox to [256, 256]
  x_center = tf.cast((search_bbox[2] - search_bbox[0]) / 2, tf.int32)
  y_center = tf.cast((search_bbox[3] - search_bbox[1]) / 2, tf.int32)
  search_final = tf.image.crop_to_bounding_box(image=search_img, offset_height=y_center - 128,
                                             offset_width=x_center - 128,
                                             target_height=256, target_width=256)
  # check size
  with tf.control_dependencies([tf.debugging.assert_equal(search_final.get_shape()[0], 256),
                                tf.debugging.assert_equal(search_final.get_shape()[1], 256),
                                tf.debugging.assert_equal(search_final.get_shape()[2], 3)]):
      search_final = tf.identity(search_final)

  ######################################## Process Score Map GT #############################################
  #[256, 256, 1], [256, 256, 1]
  score = tf.ones(shape=[16,16,1], dtype=tf.uint8)
  score = image_pad(image=score, pad_value=0, out_size=256)
  num_total = 256 * 256
  num_positive = 16 * 16
  num_negative = num_total - num_positive
  weight_positive = float(num_negative) / float(num_total)
  weight_negative = float(num_positive) / float(num_total)
  mat_positive = score * weight_positive # float
  mat_negative = (1 - score) * weight_negative # float
  score_weight = mat_positive + mat_negative
  # check size
  with tf.control_dependencies([tf.debugging.assert_equal(score.get_shape()[0], 256),
                                tf.debugging.assert_equal(score.get_shape()[1], 256),
                                tf.debugging.assert_equal(score.get_shape()[2], 1),
                                tf.debugging.assert_equal(score_weight.get_shape()[0], 256),
                                tf.debugging.assert_equal(score_weight.get_shape()[1], 256),
                                tf.debugging.assert_equal(score_weight.get_shape()[2], 1)]):
      score = tf.identity(score)
      score_weight = tf.identity(score_weight)

  ################################### Randomly flip templar/search images ####################################
  stacked = tf.stack(values=[templar_final, search_final], axis=0) # [2, 256, 256, 3]
  stacked = tf.image.random_flip_left_right(image=stacked)
  templar_final = tf.squeeze(stacked[0:1, :,:,:])
  search_final = tf.squeeze(stacked[1:2, :,:,:])

  templar_final = mean_image_subtraction(templar_final, _CHANNEL_MEANS, num_channels)
  search_final = mean_image_subtraction(search_final, _CHANNEL_MEANS, num_channels)

  return templar_final, search_final, score, score_weight


########################################################################
# Build TF Dataset of ImageNet15-VID train pipeline
########################################################################
def build_dataset(num_gpu=2, batch_size=8, train_record_dir='/storage/slurm/wangyu/imagenet15_vid/tfrecord_train',
                  is_training=True, data_format='channels_first'):
    """
    Note that though each gpu process a unique part of the dataset, the data pipeline is built
    only on CPU. Each GPU worker will have its own dataset iterator

    :param num_gpu: total number of gpus, each gpu will be assigned to a unique part of the dataset
    :param batch_size: batch of images processed on each gpu
    :param train_record_dir: where the tfrecord files are stored
    :param is_training: whether training or not
    :param data_format: must be 'channels_first' when using gpu
    :return: a list of sub-dataset for each different gpu
    """


    file_list = get_filenames(is_training, train_record_dir)
    print('Got {} tfrecords.'.format(len(file_list)))
    subset = []

    # create dataset, reading all train tfrecord files, default number of files = _NUM_SHARDS
    dataset = tf.data.Dataset.list_files(file_pattern=file_list, shuffle=False)
    # shuffle file orders
    dataset = dataset.shuffle(buffer_size=_NUM_SHARDS)
    # seperate a unique part of the dataset according to number of gpus
    for gpu_id in range(num_gpu):
        subset.append(dataset.shard(num_shards=num_gpu, index=gpu_id))
    # parse records for each gpu
    for gpu_id in range(num_gpu):
        # process 4 files concurrently and interleave blocks of 10 records from each file
        subset[gpu_id] = subset[gpu_id].interleave(lambda filename: tf.data.TFRecordDataset(filenames=filename,
                                                                                            compression_type='GZIP',
                                                                                            num_parallel_reads=4),
                                                   cycle_length=10, block_length=50, num_parallel_calls=4)
        subset[gpu_id] = subset[gpu_id].prefetch(buffer_size=batch_size*num_gpu*2) # prefetch
        subset[gpu_id] = subset[gpu_id].map(parse_func, num_parallel_calls=4) # parallel parse 4 examples at once
        if data_format == 'channels_first':
            subset[gpu_id] = subset[gpu_id].map(reformat_channel_first, num_parallel_calls=4) # parallel parse 4 examples at once
        else:
            raise ValueError('Data format is not channels_first when building dataset pipeline!')
        subset[gpu_id] = subset[gpu_id].shuffle(buffer_size=10000)
        subset[gpu_id] = subset[gpu_id].repeat()
        subset[gpu_id] = subset[gpu_id].batch(batch_size) # inference batch images for one feed-forward
        # prefetch and buffer internally, to prevent starvation of GPUs
        subset[gpu_id] = subset[gpu_id].prefetch(buffer_size=num_gpu*2)

    print('Dataset pipeline built for {} GPUs.'.format(num_gpu))

    return subset