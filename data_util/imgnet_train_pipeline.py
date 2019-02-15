'''
    The files implements ImageNet training data pipeline.
    The train image/label tfrecords are needed.

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
These steps are known as "ResNet pre-processing" which is different from "Inception pre-processing"
and "VGG pro-processing"

Images used during training are ideally sampled using the provided bounding boxes, and subsequently
cropped to the sampled bounding box. Images are additionally flipped randomly,
then resized to the target output size (without aspect-ratio preservation).
However, if there's no bounding boxes provided which is the case in this implementation, the bounding
box is assumed to be the entire image.


Images used during evaluation are resized (with aspect-ratio preservation), 
centrally cropped and undergo mean color subtraction. 

"""

import tensorflow as tf
import os

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_TRAIN = 1281167
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

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


def decode_crop_flip(image_buffer, num_channels):
    '''
    :param image_buffer:
    :return: randomly cropped and flipped image
    '''

    # the random distortion parameters are the same as used in tensorflow resnet official implementation
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        image_size=tf.image.extract_jpeg_shape(image_buffer),
        bounding_boxes=tf.reshape([], [1, 0, 4]),
        min_object_covered=0.1,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.05, 1.0],
        max_attempts=100,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, _ = sample_distorted_bounding_box

    # Reassemble the bounding box in the format the crop op requires.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

    # Use the fused decode and crop op here, which is faster than each in series.
    cropped = tf.image.decode_and_crop_jpeg(image_buffer, crop_window, channels=num_channels)

    # Flip to add a little more random distortion in.
    cropped = tf.image.random_flip_left_right(cropped)

    return cropped

def resize(image, height, width):
    '''
    :param image:
    :param height:
    :param width:
    :return: resized image without aspect-ratio preserving
    '''

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

  image = decode_crop_flip(image_buffer=image_buffer, num_channels=num_channels)
  image = resize(image, output_height, output_width)

  image.set_shape([output_height, output_width, num_channels])

  return mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)


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

  image_buffer, label = parse_example_proto(raw_record)
  image = preprocess_image(
      image_buffer=image_buffer,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label

def parse_func(example_proto):
    """
        Callable func to be fed to dataset.map()
    """

    image, label = parse_record(raw_record=example_proto,
                                is_training=True,
                                dtype=tf.float32) # image: tf.float32, [h, w, 3]; label: tf.int32, scalar
    dict = {'image': image, 'label': label}
    return dict

def reformat_channel_first(example_dict):

    image = tf.transpose(example_dict['image'], [2, 0, 1])
    dict = {'image':image, 'label':example_dict['label']}

    return dict

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""

  return [ os.path.join(data_dir, 'train_'+str(shard_id)+'.tfrecord') for shard_id in range(1024)]

########################################################################
# Build TF Dataset of ImageNet train pipeline
########################################################################
def build_dataset(num_gpu=2, batch_size=64, train_record_dir='/storage/slurm/wangyu/imagenet/tfrecord_train',
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

    # create dataset, reading all train tfrecord files, default number of files = 1024
    dataset = tf.data.Dataset.list_files(file_pattern=file_list, shuffle=False)
    # shuffle file orders
    dataset = dataset.shuffle(buffer_size=1024)
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