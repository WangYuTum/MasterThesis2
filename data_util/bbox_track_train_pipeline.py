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
_NUM_TRAIN = 0 # TODO
_NUM_SHARDS = 0 # TODO

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
    * pad boundaries to [256,256] with ImageNet mean RGB values

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
        'templar/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'templar/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'templar/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'templar/channels': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'templar/bbox/xmin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'templar/bbox/ymin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'templar/bbox/xmax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'templar/bbox/ymax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'search/channels': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/bbox/xmin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/bbox/ymin': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/bbox/xmax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'search/bbox/ymax': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'templar/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'search/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    features = tf.parse_single_example(raw_record, feature_map)
    templar_bbox = [features['templar/bbox/xmin'], features['templar/bbox/ymin'],
                    features['templar/bbox/xmax'], features['templar/bbox/ymax']]
    search_bbox = [features['search/bbox/xmin'], features['search/bbox/ymin'],
                    features['search/bbox/xmax'], features['search/bbox/ymax']]

    return features['templar/encoded'], features['search/encoded'], templar_bbox, search_bbox

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

    pad_h_total = int(out_size - img_height)
    pad_w_total = int(out_size - img_width)

    # TODO, use tf.cond for conditional
    if pad_h_total % 2 == 0:
        pad_h_begin = int(pad_h_total / 2)
        pad_h_end = int(pad_h_total / 2)
    else:
        pad_h_begin = int(pad_h_total / 2) + 1
        pad_h_end = int(pad_h_total / 2)

    if pad_w_total % 2 == 0:
        pad_w_begin = int(pad_w_total / 2)
        pad_w_end = int(pad_w_total / 2)
    else:
        pad_w_begin = int(pad_w_total / 2) + 1
        pad_w_end = int(pad_w_total / 2)

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
  p = int((bbox_h + bbox_w) / 4)
  argmin_dim = tf.math.argmin([bbox_w, bbox_h], axis=0) # 0: shorter in width, 1: shorter in height
  # TODO, check even/odd
  extend_val = tf.math.abs(bbox_w-bbox_h) / 2




  score = 0
  score_weight = 0
  # image.set_shape([_DEFAULT_SIZE, _DEFAULT_SIZE, num_channels])

  templar_img = mean_image_subtraction(templar_img, _CHANNEL_MEANS, num_channels)
  search_img = mean_image_subtraction(search_img, _CHANNEL_MEANS, num_channels)

  return templar_img, search_img, score, score_weight

'''
* Get tight bbox, randomly shift +-8 pixels
* Pad image to [1500, 1500] with mean RGB values
* Crop to 256x256:
    * get tight bbox [w, h]
    * compute context margin p = (w+h)/4
    * extend bbox to [w+2p, h+2p], and get min(w+2p, h+2p)
    * extend bbox to [D, D] by adding the shorter side with max(w+2p, h+2p) - min(w+2p, h+2p)
    * crop [D, D] and rescale to [128, 128], get the rescale factor [s]
    * pad boundaries to [256,256] with ImageNet mean RGB values
    
    
*********************************** Search image ****************************************
* Get tight bbox of the corresponding object in templar image
* Randomly rescale in range(s*0.8, s*1.2), and update bbox position; [s] is computed during pre-process templar image
* Pad image to [1500, 1500] with mean RGB values
* Set bbox as the center and crop the image to [256, 256] so that search target is centered in the image
'''
