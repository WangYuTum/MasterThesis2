'''
    The files implements ImageNet validation data pipeline.
    The validation image/label tfrecords are needed.

    Some functions/methods are directly copied from tensorflow official resnet model:
    https://github.com/tensorflow/models/tree/r1.12.0/official/resnet

    Therefore the code must be used under Apache License, Version 2.0 (the "License"):
    http://www.apache.org/licenses/LICENSE-2.0
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


"""
Note that pre-processing validation images are different than pre-processing training images.
Images used during evaluation are resized (with aspect-ratio preservation) and
centrally cropped. All images undergo mean color subtraction. 
These steps are known as "ResNet pre-processing" which is different from "Inception pre-processing"
and "VGG pro-processing"
"""

import tensorflow as tf
import sys, os, glob

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_VAL = 50000
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# The lower bound for the smallest side of the image for aspect-preserving
# resizing. For example, if an image is 500 x 1000, it will be resized to
# _RESIZE_MIN x (_RESIZE_MIN * 2).
_RESIZE_MIN = 256


def central_crop(image, crop_height, crop_width):
  """Performs central crops of the given image list.
  Args:
    image: a 3-D image tensor
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.
  Returns:
    3-D tensor with cropped image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  amount_to_be_cropped_h = (height - crop_height)
  crop_top = amount_to_be_cropped_h // 2
  amount_to_be_cropped_w = (width - crop_width)
  crop_left = amount_to_be_cropped_w // 2
  return tf.slice(
      image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def mean_image_subtraction(image, means, num_channels):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
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


def smallest_size_at_least(height, width, resize_min):
  """Computes new shape with the smallest side equal to `smallest_side`.
  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.
  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: an int32 scalar tensor indicating the new width.
  """
  resize_min = tf.cast(resize_min, tf.float32)

  # Convert to floats to make subsequent calculations go smoothly.
  height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

  smaller_dim = tf.minimum(height, width)
  scale_ratio = resize_min / smaller_dim

  # Convert back to ints to make heights and widths that TF ops will accept.
  new_height = tf.cast(height * scale_ratio, tf.int32)
  new_width = tf.cast(width * scale_ratio, tf.int32)

  return new_height, new_width

def aspect_preserving_resize(image, resize_min):
  """Resize images preserving the original aspect ratio.
  Args:
    image: A 3-D image `Tensor`.
    resize_min: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.
  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  shape = tf.shape(image)
  height, width = shape[0], shape[1]

  new_height, new_width = smallest_size_at_least(height, width, resize_min)

  return resize_image(image, new_height, new_width)


def resize_image(image, height, width):
  """Simple wrapper around tf.resize_images.
  This is primarily to make sure we use the same `ResizeMethod` and other
  details each time.
  Args:
    image: A 3-D image `Tensor`.
    height: The target height for the resized image.
    width: The target width for the resized image.
  Returns:
    resized_image: A 3-D tensor containing the resized image. The first two
      dimensions have the shape [height, width].
  """
  return tf.image.resize_images(
      image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)

def preprocess_image(image_buffer, output_height, output_width,
                     num_channels, is_training=False):
  """Preprocesses the given image.
  Preprocessing includes decoding, cropping, and resizing for both training
  and eval images. Training preprocessing, however, introduces some random
  distortion of the image to improve accuracy.
  Args:
    image_buffer: scalar string Tensor representing the raw JPEG image buffer.
    output_height: The height of the image after pre-processing.
    output_width: The width of the image after pre-processing.
    num_channels: Integer depth of the image buffer for decoding.
    is_training: `True` if we're pre-processing the image for training and `False` otherwise.

  Returns:
    A preprocessed image. [height, width, channels]
  """

  if is_training:
      raise ValueError('Only support validation!')

  image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
  image = aspect_preserving_resize(image, _RESIZE_MIN)
  image = central_crop(image, output_height, output_width)

  image.set_shape([output_height, output_width, num_channels])

  return mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
      raise  ValueError('Only support validation!')
  else:
    return [
        os.path.join(data_dir, 'val_'+str(shard_id)+'.tfrecord')
        for shard_id in range(128)]

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.
  The input record is parsed into a label and image, and the image is passed
  through pre-processing steps.
  Args:
    raw_record: scalar Tensor tf.string containing a serialized Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.
  Returns:
    Tuple with processed image tensor and label id.
  """
  if is_training:
      raise ValueError('Only support validation!')

  image_buffer, label = parse_example_proto(raw_record)
  image = preprocess_image(
      image_buffer=image_buffer,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label

def parse_example_proto(raw_record):
    '''
    :param raw_record: scalar Tensor tf.string containing a serialized Example protocol buffer.
    :return:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
    '''

    feature_map = {
        'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/colorspace': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/channels': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/class/label': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/format': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
    }
    features = tf.parse_single_example(raw_record, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)

    return features['image/encoded'], label