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
from scipy.ndimage.morphology import binary_dilation

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
* Get tight bbox
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
* Put bbox in the center, randomly shift bbox +-64 away from center and get new center
* crop the image to [256, 256] so that shifted search target is centered in the image


*********************************** Loc GT ****************************************
* Make a zeros mask [17, 17], tf.int32
* Set the pixels to ones where the target is located, radius <= 16 is considered as positive
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

    dict = {'templar':templar, 'search':search, 'score':score, 'score_weight':score_weight,
            'tight_temp_bbox': example_dict['tight_temp_bbox'], 'tight_search_bbox': example_dict['tight_search_bbox']}

    return dict

def parse_func(example_proto):
    """
        Callable func to be fed to dataset.map()
    """

    parsed_dict = parse_record(raw_record=example_proto,
                               is_training=True,
                               dtype=tf.float32) # images/score_weight: tf.float32, [h, w, 3]; score: tf.int32

    return parsed_dict

def filter_bbox(parsed_dict):
    '''
    :param parsed_dict:
    :return: true if examples hold the condition
    '''
    # filter the examples whose tight bbox height or width <4
    # [xmin, ymin, xmax, ymax]

    temp_tight_bbox = parsed_dict['tight_temp_bbox']
    search_tight_bbox = parsed_dict['tight_search_bbox']
    keep_bool0 = tf.logical_and(tf.greater_equal(temp_tight_bbox[2] - temp_tight_bbox[0], 4),
                                tf.greater_equal(temp_tight_bbox[3] - temp_tight_bbox[1], 4))
    keep_bool1 = tf.logical_and(tf.greater_equal(search_tight_bbox[2] - search_tight_bbox[0], 4),
                                tf.greater_equal(search_tight_bbox[3] - search_tight_bbox[1], 4))
    keep_bool = tf.logical_and(keep_bool0, keep_bool1)

    return keep_bool

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
    templar_img, search_img, score, score_weight, tight_temp_bbox, tight_search_bbox = preprocess_pair(
        templar_buffer=templar_buffer, search_buffer=search_buffer, templar_bbox=templar_bbox,
        search_bbox=search_bbox, num_channels=_NUM_CHANNELS, is_training=is_training)

    templar_img = tf.cast(templar_img, dtype)
    search_img = tf.cast(search_img, dtype)
    score = tf.cast(score, tf.int32)
    score_weight = tf.cast(score_weight, dtype)
    tight_temp_bbox = tf.cast(tight_temp_bbox, tf.int32)
    #tight_search_bbox = tf.cast(tight_search_bbox, tf.int32)

    dict = {'templar': templar_img, 'search': search_img, 'score': score, 'score_weight': score_weight,
            'tight_temp_bbox': tight_temp_bbox, 'tight_search_bbox': tight_search_bbox}

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
    templar_img, templar_box, search_img, search_box = tf.cond(tf.random.uniform([]) < 0.5, true_fn=true_fn, false_fn=false_fn)

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
    :return: padded image of shape [out_size, out_size, 3], new bbox
    '''

    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0], instead of {}'.format(image.get_shape().ndims))
    #if image.dtype != tf.uint8:
    #    raise ValueError('Input must be of type tf.uint8')
    img_height = tf.shape(image)[0]
    img_width = tf.shape(image)[1]

    pad_h_total = tf.cast(out_size - img_height, tf.int32)
    pad_w_total = tf.cast(out_size - img_width, tf.int32)

    pad_h_cond = tf.equal(pad_h_total % 2, 0)
    pad_h_begin = tf.cond(pad_h_cond,
                          lambda: tf.cast(pad_h_total / 2, tf.int32),
                          lambda: tf.cast(pad_h_total / 2, tf.int32) + 1)
    pad_h_end = tf.cond(pad_h_cond,
                        lambda: tf.cast(pad_h_total / 2, tf.int32),
                        lambda: tf.cast(pad_h_total / 2, tf.int32))

    pad_w_cond = tf.equal(pad_w_total % 2, 0)
    pad_w_begin = tf.cond(pad_w_cond,
                          lambda: tf.cast(pad_w_total / 2, tf.int32),
                          lambda: tf.cast(pad_w_total / 2, tf.int32) + 1)
    pad_w_end = tf.cond(pad_w_cond,
                        lambda: tf.cast(pad_w_total / 2, tf.int32),
                        lambda: tf.cast(pad_w_total / 2, tf.int32))

    # check if pad values are <=0, don't pad
    no_pad_h = tf.less_equal(pad_h_total, 0)
    no_pad_w = tf.less_equal(pad_w_total, 0)
    def return_zeros(): return 0,0
    def return_identity(x, y): return x, y
    pad_h_begin, pad_h_end = tf.cond(no_pad_h, return_zeros, lambda: return_identity(pad_h_begin, pad_h_end))
    pad_w_begin, pad_w_end = tf.cond(no_pad_w, return_zeros, lambda: return_identity(pad_w_begin, pad_w_end))

    # do padding
    image_padded = tf.pad(tensor=image, paddings=[[pad_h_begin, pad_h_end],[pad_w_begin, pad_w_end],[0,0]],
                          mode='CONSTANT', name=None, constant_values=0)

    return image_padded, pad_h_begin, pad_w_begin


def distort_bounding_box(input_bbox, random_shift):
    '''
    :param input_bbox: [xmin, ymin, xmax, ymax]
    :param random_shift: integer
    :return: [xmin', ymin', xmax', ymax']
    '''

    h_rand = tf.random.uniform(shape=[], minval=-(random_shift-1), maxval=random_shift, dtype=tf.int32)
    w_rand = tf.random.uniform(shape=[], minval=-(random_shift-1), maxval=random_shift, dtype=tf.int32)

    return [input_bbox[0]+w_rand, input_bbox[1]+h_rand, input_bbox[2]+w_rand, input_bbox[3]+h_rand], h_rand, w_rand

def extend_bbox_w(templar_bbox, extend_val_left, extend_val_right):
    '''
    :param templar_bbox: list [xmin, ymin, xmax, ymax], int
    :param extend_val_left:
    :param extend_val_right:
    :return:
    '''
    #extend_val_left = tf.cast(extend_val_left, tf.int32)
    #extend_val_right = tf.cast(extend_val_right, tf.int32)
    # ymin, ymax stay the same
    return [templar_bbox[0]-extend_val_left, templar_bbox[1], templar_bbox[2]+extend_val_right, templar_bbox[3]]

def extend_bbox_h(templar_bbox, extend_val_left, extend_val_right):
    '''
    :param templar_bbox: list [xmin, ymin, xmax, ymax], int
    :param extend_val_left:
    :param extend_val_right:
    :return:
    '''
    #extend_val_left = tf.cast(extend_val_left, tf.int32)
    #extend_val_right = tf.cast(extend_val_right, tf.int32)
    # xmin, xmax stay the same
    return [templar_bbox[0], templar_bbox[1]-extend_val_left, templar_bbox[2], templar_bbox[3]+extend_val_right]

def rescale_bbox(search_bbox, rescale_factor):

    new_bbox = []
    for i in range(4):
        item = search_bbox[i]
        item = tf.cast(item, tf.float32) * rescale_factor
        new_bbox.append(tf.cast(item, tf.int32))

    return new_bbox

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
    templar_bbox = tf.cast(templar_bbox, tf.int32)
    search_bbox = tf.cast(search_bbox, tf.int32)

    def return_zero_pad(x): return [0, tf.abs(x)]
    def return_iden_no_pad(x): return [x, 0]
    def return_maxW_pad(x, w_max): return [w_max - 1, x - (w_max - 1)]
    def return_maxH_pad(x, h_max): return [h_max - 1, x - (h_max - 1)]
    def flip_bbox(bbox, img_w):
        '''
        :param bbox: original bbox [xmin, ymin, xmax, ymax]
        :param img_w:
        :return: flipped bbox
        '''
        new_bbox = []
        new_bbox.append(img_w - bbox[2])
        new_bbox.append(bbox[1])
        new_bbox.append(img_w - bbox[0])
        new_bbox.append(bbox[3])

        return new_bbox

    ######################################## Process Templar #############################################
    # Get tight bbox, always keep the target at the center
    #templar_bbox = distort_bounding_box(input_bbox=templar_bbox, random_shift=8) # new box [xmin, ymin, xmax, ymax]
    # pad border in case distorted bbox out of boundary
    mean_rgb = tf.reduce_mean(tf.cast(templar_img, tf.int64)) # tf.uint8
    mean_rgb = tf.cast(mean_rgb, tf.uint8)
    #templar_img = templar_img - mean_rgb
    #pad_border, pad_border = 10, 10
    #templar_img = tf.pad(tensor=templar_img, paddings=[[pad_border, pad_border], [pad_border, pad_border],[0, 0]],
    #                     mode='CONSTANT', name=None, constant_values=0)
    #templar_img = templar_img + mean_rgb
    # update tight bbox position, the size stays the same, the 4 corners are updated
    #templar_bbox[0] = templar_bbox[0] + pad_border
    #templar_bbox[1] = templar_bbox[1] + pad_border
    #templar_bbox[2] = templar_bbox[2] + pad_border
    #templar_bbox[3] = templar_bbox[3] + pad_border
    bbox_h = templar_bbox[3] - templar_bbox[1]
    bbox_w = templar_bbox[2] - templar_bbox[0]
    # save the (distorted) tight bbox for display
    tight_bbox = []
    tight_bbox.append(templar_bbox[0])
    tight_bbox.append(templar_bbox[1])
    tight_bbox.append(templar_bbox[2])
    tight_bbox.append(templar_bbox[3])
    p = tf.cast((bbox_h + bbox_w) / 4, tf.int32) # get context margin and compute new bbox
    argmin_dim = tf.math.argmin([bbox_w, bbox_h], axis=0) # 0: shorter in width, 1: shorter in height
    extend_w_cond = tf.equal(argmin_dim, 0) # true if extend in width dim, otherwise extend in height dim
    extend_side_cond = tf.equal(tf.math.abs(bbox_w-bbox_h) % 2, 0) # if true, extend evenly on both side
    extend_val_left = tf.cond(extend_side_cond,
                              lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32),
                              lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32) + 1)
    extend_val_right = tf.cast(tf.math.abs(bbox_w-bbox_h) / 2, tf.int32)
    # get a rect bbox by extending the shorter side
    templar_bbox_new = tf.cond(extend_w_cond, lambda: extend_bbox_w(templar_bbox, extend_val_left, extend_val_right),
                               lambda: extend_bbox_h(templar_bbox, extend_val_left, extend_val_right))
    ## add context margin
    templar_bbox_new = [templar_bbox_new[0]-p, templar_bbox_new[1]-p, templar_bbox_new[2]+p, templar_bbox_new[3]+p]
    tight_bbox[0] = tight_bbox[0] - templar_bbox_new[0] # [xmin, ymin, xmax, ymax]
    tight_bbox[1] = tight_bbox[1] - templar_bbox_new[1]
    tight_bbox[2] = tight_bbox[2] - templar_bbox_new[0]
    tight_bbox[3] = tight_bbox[3] - templar_bbox_new[1]
    # here the rectangular bbox might already out of boundary, must pad precise number of pixels on left/up
    img_height = tf.shape(templar_img)[0]
    img_width = tf.shape(templar_img)[1]
    [new_x_min, pad_w_begin] = tf.cond(templar_bbox_new[0] < 0, lambda :return_zero_pad(templar_bbox_new[0]), lambda :return_iden_no_pad(templar_bbox_new[0]))
    [new_x_max, pad_w_end] = tf.cond(templar_bbox_new[2] >= img_width, lambda :return_maxW_pad(templar_bbox_new[2], img_width), lambda :return_iden_no_pad(templar_bbox_new[2]))
    [new_y_min, pad_h_begin] = tf.cond(templar_bbox_new[1] < 0, lambda :return_zero_pad(templar_bbox_new[1]), lambda :return_iden_no_pad(templar_bbox_new[1]))
    [new_y_max, pad_h_end] = tf.cond(templar_bbox_new[3] >= img_height, lambda :return_maxH_pad(templar_bbox_new[3], img_height), lambda :return_iden_no_pad(templar_bbox_new[3]))
    # do paddings, only effective if out of boundary
    templar_img = templar_img - mean_rgb
    templar_img = tf.pad(tensor=templar_img,
                         paddings=[[pad_h_begin, pad_h_end + 10], [pad_w_begin, pad_w_end + 10], [0, 0]],
                         mode='CONSTANT', name=None, constant_values=0)
    templar_img = templar_img + mean_rgb
    # crop the image
    croped_templar = tf.image.crop_to_bounding_box(image=templar_img, offset_height=new_y_min,
                                                   offset_width=new_x_min,
                                                   target_height=templar_bbox_new[3]-templar_bbox_new[1],
                                                   target_width=templar_bbox_new[2]-templar_bbox_new[0])
    with tf.control_dependencies([tf.debugging.assert_equal(templar_bbox_new[3] - templar_bbox_new[1],
                                                            templar_bbox_new[2] - templar_bbox_new[0])]):
        # rescale to [127, 127], get the scale factor
        scale_s = 127.0 / tf.cast(templar_bbox_new[3] - templar_bbox_new[1], tf.float32)
        # rescale the tight bbox
        tight_temp_bbox = rescale_bbox(tight_bbox, scale_s)
        scale_s = tf.debugging.assert_all_finite(t=scale_s, msg='scale factor not a number!')
        croped_templar = tf.image.resize_bilinear(images=tf.expand_dims(croped_templar, axis=0), size=[127, 127])
        croped_templar = tf.squeeze(croped_templar, axis=0) # [h, w, 3]
    # check size
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(croped_templar)[0], 127),
                                  tf.debugging.assert_equal(tf.shape(croped_templar)[1], 127),
                                  tf.debugging.assert_equal(tf.shape(croped_templar)[2], 3)]):
        templar_final = tf.identity(croped_templar)

    ######################################## Process Search image #############################################
    # Get rgb mean
    mean_rgb = tf.reduce_mean(tf.cast(search_img, tf.int64))  # tf.uint8
    mean_rgb = tf.cast(mean_rgb, tf.float32)
    # Get random scale factor
    rescale_factor = scale_s * float(np.random.randint(low=8, high=13, size=None, dtype=np.int32)) / 10.0
    rescale_factor = tf.debugging.assert_all_finite(t=rescale_factor, msg='rescale_factor factor not a number!')
    # Get rescaled bbox position, and the image
    search_bbox = rescale_bbox(search_bbox, rescale_factor)
    new_height = tf.cast(tf.cast(tf.shape(search_img)[0], tf.float32) * rescale_factor, tf.int32)
    new_width = tf.cast(tf.cast(tf.shape(search_img)[1], tf.float32) * rescale_factor, tf.int32)
    search_img = tf.image.resize_bilinear(images=tf.expand_dims(search_img, axis=0), size=[new_height, new_width])
    search_img = tf.squeeze(search_img, axis=0)  # [h, w, 3]
    ### randomly shift bbox +-64 pixels, get the shift values and new bbox center
    search_bbox, h_shift, w_shift = distort_bounding_box(input_bbox=search_bbox, random_shift=64)  # new box [xmin, ymin, xmax, ymax], h_shift, w_shift
    ### crop around the center of the bbox to [255, 255], if out of boundary, pad with mean rgb value
    img_width = tf.shape(search_img)[1]
    img_height = tf.shape(search_img)[0]
    x_center = tf.cast((search_bbox[2] - search_bbox[0]) / 2, tf.int32) + search_bbox[0]
    y_center = tf.cast((search_bbox[3] - search_bbox[1]) / 2, tf.int32) + search_bbox[1]
    x_min, x_max = x_center - 127, x_center + 127
    y_min, y_max = y_center - 127, y_center + 127
    [new_x_min, pad_w_begin] = tf.cond(x_min < 0, lambda :return_zero_pad(x_min), lambda :return_iden_no_pad(x_min))
    [new_x_max, pad_w_end] = tf.cond(x_max >= img_width, lambda :return_maxW_pad(x_max, img_width), lambda :return_iden_no_pad(x_max))
    [new_y_min, pad_h_begin] = tf.cond(y_min < 0, lambda :return_zero_pad(y_min), lambda :return_iden_no_pad(y_min))
    [new_y_max, pad_h_end] = tf.cond(y_max >= img_height, lambda :return_maxH_pad(y_max, img_height), lambda :return_iden_no_pad(y_max))
    # do paddings, only effective if out of boundary
    search_img = search_img - mean_rgb
    search_img = tf.pad(tensor=search_img, paddings=[[pad_h_begin, pad_h_end+10], [pad_w_begin, pad_w_end+10], [0, 0]],
                        mode='CONSTANT', name=None, constant_values=0)
    search_img = search_img + mean_rgb
    # crop
    search_final = tf.image.crop_to_bounding_box(image=search_img, offset_height=new_y_min, offset_width=new_x_min,
                                                 target_height=255, target_width=255)
    ## get tight bbox within the rescaled search img [xmin, ymin, xmax, ymax]
    bbox_h_half = tf.cast((search_bbox[3] - search_bbox[1]) / 2, tf.int32) # might be zero
    bbox_w_half = tf.cast((search_bbox[2] - search_bbox[0]) / 2, tf.int32) # might be zero
    tight_search_bbox = []
    tight_search_bbox.append(127 - bbox_w_half - w_shift) # xmin
    tight_search_bbox.append(127 - bbox_h_half - h_shift) # ymin
    tight_search_bbox.append(127 + bbox_w_half - w_shift) # xmax
    tight_search_bbox.append(127 + bbox_h_half - h_shift) # ymax
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(search_final)[0], 255),
                                  tf.debugging.assert_equal(tf.shape(search_final)[1], 255),
                                  tf.debugging.assert_equal(tf.shape(search_final)[2], 3)]):
        search_final = tf.identity(search_final)

    ######################################## Process Score Map GT #############################################
    # [17, 17, 1], [17, 17, 1]
    # consider 8 x (center - offset) <= 16 as positives, stride=8; also note that target in search image is already shifted
    t_center_x = tf.cast(8.0 - tf.cast(w_shift, tf.float32) / 8.0, tf.int32)
    t_center_y = tf.cast(8.0 - tf.cast(h_shift, tf.float32) / 8.0, tf.int32)
    score, score_weight = tf.py_func(func=build_gt_py, inp=[t_center_x, t_center_y], Tout=[tf.int32, tf.float32],
                                     stateful=True, name=None)
    """
    score = tf.zeros([17, 17, 1], dtype=tf.int32)
    delta = tf.sparse.SparseTensor(indices=[[t_center_y, t_center_x, 0]], values=[1], dense_shape=[17,17,1])
    score = score + tf.sparse.to_dense(delta)
    score = tf.expand_dims(score, axis=0) # [1,17,17,1]
    dila_structure = np.array([[False, False, True, False, False],
                               [False, True, True, True, False],
                               [True, True, True, True, True],
                               [False, True, True, True, False],
                               [False, False, True, False, False]], dtype=bool)
    dila_structure = dila_structure.astype(np.int32)
    dila_structure = np.expand_dims(dila_structure, axis=-1) # [5,5,1]
    score = tf.nn.dilation2d(input=score, filter=dila_structure, strides=[1,1,1,1], rates=[1,1,1,1], padding='SAME')
    num_total = 17 * 17
    num_positive = tf.reduce_sum(score)
    num_negative = num_total - num_positive
    weight_positive = tf.cast(num_negative, tf.float32) / tf.cast(num_total, tf.float32)
    weight_negative = tf.cast(num_positive, tf.float32) / tf.cast(num_total, tf.float32)
    mat_positive = tf.cast(score, tf.float32) * weight_positive # float
    mat_negative = (1.0 - tf.cast(score, tf.float32)) * weight_negative # float
    score_weight = mat_positive + mat_negative
    score = tf.squeeze(score, 0)
    score_weight = tf.squeeze(score_weight, 0)
    """
    # check size
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(score)[0], 17),
                                  tf.debugging.assert_equal(tf.shape(score)[1], 17),
                                  tf.debugging.assert_equal(tf.shape(score)[2], 1),
                                  tf.debugging.assert_equal(tf.shape(score_weight)[0], 17),
                                  tf.debugging.assert_equal(tf.shape(score_weight)[1], 17),
                                  tf.debugging.assert_equal(tf.shape(score_weight)[2], 1)]):
        score = tf.identity(score)
        score_weight = tf.identity(score_weight)

    ################################### Randomly flip templar/search images ####################################
    flip_v = tf.random.uniform(shape=[]) # scalar
    flip_v = tf.greater_equal(flip_v, 0.5)
    templar_final = tf.cond(flip_v, lambda : tf.image.flip_left_right(image=templar_final), lambda :templar_final)
    search_final = tf.cond(flip_v, lambda: tf.image.flip_left_right(image=search_final), lambda: search_final)
    score = tf.cond(flip_v, lambda :tf.image.flip_left_right(image=score), lambda :score)
    score_weight = tf.cond(flip_v, lambda :tf.image.flip_left_right(image=score_weight), lambda :score_weight)
    tight_search_bbox = tf.cond(flip_v, lambda :flip_bbox(tight_search_bbox, 255), lambda :tight_search_bbox)

    templar_final = mean_image_subtraction(templar_final, _CHANNEL_MEANS, num_channels)
    search_final = mean_image_subtraction(search_final, _CHANNEL_MEANS, num_channels)

    return templar_final, search_final, score, score_weight, tight_temp_bbox, tight_search_bbox

def build_gt_py(t_center_x, t_center_y):
    '''
    :param t_center_x: scalar, int32
    :param t_center_y: scalar, int32
    :return: numpy array: score, score_weight
    '''

    score = np.zeros((17,17), dtype=np.int32)
    score[t_center_y, t_center_x] = 1
    dila_structure = np.array([[False, False, True, False, False],
                               [False, True, True, True, False],
                               [True, True, True, True, True],
                               [False, True, True, True, False],
                               [False, False, True, False, False]], dtype=bool)
    dilated_score = binary_dilation(input=score, structure=dila_structure).astype(np.int32) # [17,17]
    num_total = 17 * 17
    num_positive = np.sum(dilated_score)
    num_negative = num_total - num_positive
    weight_positive = num_negative.astype(np.float32) / float(num_total)
    weight_negative = num_positive.astype(np.float32) / float(num_total)
    mat_positive = score.astype(np.float32) * weight_positive  # float
    mat_negative = (1.0 - score.astype(np.float32)) * weight_negative  # float
    score_weight = mat_positive + mat_negative

    return np.expand_dims(dilated_score, axis=-1), np.expand_dims(score_weight, axis=-1)


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
                                                   cycle_length=10, block_length=10, num_parallel_calls=4)
        subset[gpu_id] = subset[gpu_id].prefetch(buffer_size=batch_size*2) # prefetch
        subset[gpu_id] = subset[gpu_id].map(parse_func, num_parallel_calls=4) # parallel parse 4 examples at once
        subset[gpu_id] = subset[gpu_id].filter(filter_bbox) # filter tiny tight bbox
        if data_format == 'channels_first':
            subset[gpu_id] = subset[gpu_id].map(reformat_channel_first, num_parallel_calls=4) # parallel parse 4 examples at once
        else:
            raise ValueError('Data format is not channels_first when building dataset pipeline!')
        subset[gpu_id] = subset[gpu_id].shuffle(buffer_size=3000)
        subset[gpu_id] = subset[gpu_id].repeat()
        subset[gpu_id] = subset[gpu_id].batch(batch_size) # inference batch images for one feed-forward
        # prefetch and buffer internally, to prevent starvation of GPUs
        subset[gpu_id] = subset[gpu_id].prefetch(buffer_size=batch_size*2)

    print('Dataset pipeline built for {} GPUs.'.format(num_gpu))

    return subset