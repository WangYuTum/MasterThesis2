'''
    The file implements cross-correlation bbox tracking and mask propagation train data pipeline.

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
import os, sys
from scipy.ndimage.morphology import binary_dilation

_NUM_CHANNELS = 3
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_NUM_TRAIN = 272459 # number of valid training pairs: 272459, num total pairs: 285849
_NUM_SHARDS = 256 # number of tfrecords in total, each tfrecord has 1116 or 1269 pairs

"""
The tfrecord files must provide the following contents, each example have:
* A pair of images (templar image/mask, search image/mask)
* Object ID of the mask
* Height, width of templar/search images 

Pre-processing of training pairs for bbox tracking:
*********************************** Templar image ****************************************
* Get tight bbox and mask of an object id
* Crop to 127x127:
    * get tight bbox [w, h]
    * compute context margin p = (w+h)/4
    * extend bbox to [w+2p, h+2p], and get min(w+2p, h+2p)
    * extend bbox to [D, D] by adding the shorter side with max(w+2p, h+2p) - min(w+2p, h+2p)
    * crop img/mask [D, D] and rescale to [127, 127], get the rescale factor [s]
    * pad with mean RGB values if exceed boundary
    * stack RGB img and mask together

*********************************** Search image ****************************************
* Get tight bbox and mask of the corresponding object in templar image
* Randomly rescale in range(s*0.8, s*1.2), and update bbox/mask; [s] is computed during pre-process templar image
* Put bbox/mask in the center, randomly shift bbox +-32 away from center and get new center
* crop the image/mask to [255, 255] so that shifted search target is centered in the image
* generate a gaussian mask centered at image center
* stack RGB img and Gaussian mask together


*********************************** Loc GT ****************************************
* Make a zeros mask [17, 17], tf.int32
* Set the pixels to ones where the target is located, radius <= 16 is considered as positive
* Make balanced weight mask for the GT

*********************************** Mask GT ****************************************
* get the gt mask from shifted search img as [127, 127]
* for each positive position, get the gt mask by shifting a multiple of 8 pixels accordingly, 
there should be 13 masks in total; each mask has size [127, 127], which is the same size as the templar image
* stack masks together(keep the order: left to right, top to bottom)


The above pre-processing preserve aspect-ratio.
Images/Masks are additionally flipped randomly (templar & search flip together).
Images undergo mean color subtraction from ImageNet12.
"""

def get_filenames(data_dir):
    """Return filenames for dataset."""

    return [os.path.join(data_dir, 'train_'+str(shard_id)+'.tfrecord') for shard_id in range(_NUM_SHARDS)]

def parse_func(parsed_dict):
    """
        Callable func to be fed to dataset.map()
    """

    parsed_dict = parse_record(parsed_dict=parsed_dict,
                               dtype=tf.float32) # images/score_weight: tf.float32, [h, w, 3]; score: tf.int32

    return parsed_dict

def parse_example_proto(raw_record):
    '''
    :param raw_record: scalar Tensor tf.string containing a serialized Example protocol buffer.
    :return:
        templar_buffer: Tensor tf.string containing the contents of a JPEG file.
        search_buffer: Tensor tf.string containing the contents of a JPEG file.
    '''

    feature_map = {
        'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image/object_id': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
        'image0/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'image1/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'anno0/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        'anno1/encoded': tf.FixedLenFeature([], dtype=tf.string, default_value='')}
    features = tf.parse_single_example(raw_record, feature_map)
    img0_buffer = features['image0/encoded']
    img1_buffer = features['image1/encoded']
    mask0_buffer = features['anno0/encoded']
    mask1_buffer = features['anno1/encoded']
    object_id = features['image/object_id']
    img_h = features['image/height']
    img_w = features['image/width']

    # randomly select one image/bbox as templar
    def true_fn(): return img0_buffer, mask0_buffer, img1_buffer, mask1_buffer
    def false_fn(): return img1_buffer, mask1_buffer, img0_buffer, mask0_buffer
    templar_img, templar_mask, search_img, search_mask = tf.cond(tf.random.uniform([]) < 0.5, true_fn=true_fn, false_fn=false_fn)

    return templar_img, search_img, templar_mask, search_mask, object_id, img_h, img_w

def parse_raw(example_proto):

    # parse raw buffers
    templar_buffer, search_buffer, templar_mask_buffer, search_mask_buffer, object_id, img_h, img_w = parse_example_proto(example_proto)
    # decode buffers
    # decode raw buffers
    templar_img = tf.image.decode_jpeg(templar_buffer, channels=3)  # uint8, [h,w,3]
    search_img = tf.image.decode_jpeg(search_buffer, channels=3)  # uint8, [h,w,3]
    templar_mask = tf.image.decode_png(templar_mask_buffer, channels=1)  # uint8, [h,w,1]
    search_mask = tf.image.decode_png(search_mask_buffer, channels=1)  # uint8, [h,w,1]
    object_id = tf.cast(object_id, tf.int32)
    img_h = tf.cast(img_h, tf.int32)
    img_w = tf.cast(img_w, tf.int32)


    dict = {'templar_img': templar_img, 'search_img': search_img, 'templar_mask': templar_mask,
            'search_mask': search_mask, 'object_id': object_id, 'img_h': img_h, 'img_w': img_w}

    return dict

def filter_empty(parsed_dict):

    # get object id
    object_id = tf.cast(parsed_dict['object_id'], tf.uint8) # tf.uint8

    # get all possible ids in mask
    [unique_values0, _] = tf.unique(x=tf.reshape(parsed_dict['templar_mask'], [-1])) # tf.uint8
    bool0 = tf.math.reduce_any(tf.math.equal(unique_values0, object_id))

    [unique_values1, _] = tf.unique(x=tf.reshape(parsed_dict['search_mask'], [-1]))  # tf.uint8
    bool1 = tf.math.reduce_any(tf.math.equal(unique_values1, object_id))

    keep_bool = tf.math.logical_and(bool0, bool1)

    return keep_bool

def parse_record(parsed_dict, dtype):
    """Parses a record containing a training example of templar/search image pair.
    The image buffers are passed to be pre-processed
    Args:
    raw_record: scalar Tensor tf.string containing a serialized Example protocol buffer.
    dtype: data type to use for images/features.
    Returns:
    Parsed example of dict type: {'templar':templar, 'search':search, 'score':score, 'score_weight':score_weight}
    """

    templar_img_mask, search_img_mask, score, score_weight, gt_masks, tight_temp_bbox, tight_search_bbox, \
    gt_masks_weight= preprocess_pair(
        templar_img=parsed_dict['templar_img'], search_img=parsed_dict['search_img'],
        templar_mask=parsed_dict['templar_mask'], search_mask=parsed_dict['search_mask'],
        object_id=parsed_dict['object_id'], num_channels=_NUM_CHANNELS,
        img_h=parsed_dict['img_h'], img_w=parsed_dict['img_w'])

    templar_img_mask = tf.cast(templar_img_mask, dtype)
    search_img_mask = tf.cast(search_img_mask, dtype)
    score = tf.cast(score, tf.int32)
    score_weight = tf.cast(score_weight, dtype)
    gt_masks_weight = tf.cast(gt_masks_weight, dtype)
    gt_masks = tf.cast(gt_masks, tf.int32)
    tight_temp_bbox = tf.cast(tight_temp_bbox, tf.int32)
    tight_search_bbox = tf.cast(tight_search_bbox, tf.int32)

    dict = {'templar_img_mask': templar_img_mask, 'search_img_mask': search_img_mask, 'score': score,
            'score_weight': score_weight, 'gt_masks': gt_masks, 'tight_temp_bbox': tight_temp_bbox,
            'tight_search_bbox': tight_search_bbox, 'gt_masks_weight': gt_masks_weight}

    return dict

def rescale_bbox(bbox, rescale_factor):

    new_bbox = []
    for i in range(4):
        item = bbox[i]
        item = tf.cast(item, tf.float32) * rescale_factor
        new_bbox.append(tf.cast(item, tf.int32))

    return new_bbox

def extend_bbox_w(templar_bbox, extend_val_left, extend_val_right):

    # ymin, ymax stay the same
    return [templar_bbox[0]-extend_val_left, templar_bbox[1], templar_bbox[2]+extend_val_right, templar_bbox[3]]

def extend_bbox_h(templar_bbox, extend_val_left, extend_val_right):

    # xmin, xmax stay the same
    return [templar_bbox[0], templar_bbox[1]-extend_val_left, templar_bbox[2], templar_bbox[3]+extend_val_right]

def bbox_from_mask(mask):
    '''
    :param mask: [h, w, 1], tf.uint8/tf.float32
    :return:
        tight bbox as a list [xmin, ymin, xmax, ymax], tf.int32
    '''

    # convert to float mask and binarize mask
    f_mask = tf.cast(mask, tf.float32)
    bin_mask = tf.cast(tf.math.greater_equal(f_mask, 0.5), tf.uint8)

    zero_t = tf.constant(0, dtype=tf.uint8)
    bool_mat = tf.math.not_equal(x=tf.squeeze(bin_mask, axis=-1), y=zero_t, name=None) # [h, w]
    indices = tf.where(bool_mat) # [n, 2], n is number of non-zeros, the other dim gives the index
    max_v = tf.math.reduce_max(indices, axis=[0], keepdims=False) # (max_h, max_w)
    min_v = tf.math.reduce_min(indices, axis=[0], keepdims=False) # (min_h, min_w)
    max_h = tf.cast(max_v[0], tf.int32)
    max_w = tf.cast(max_v[1], tf.int32)
    min_h = tf.cast(min_v[0], tf.int32)
    min_w = tf.cast(min_v[1], tf.int32)

    return [min_w, min_h, max_w+1, max_h+1] # the xmax, ymax are exclusive

def distort_bounding_box(input_bbox, random_shift):
    '''
    :param input_bbox: [xmin, ymin, xmax, ymax]
    :param random_shift: integer
    :return: [xmin', ymin', xmax', ymax']
    '''

    h_rand = tf.random.uniform(shape=[], minval=-random_shift, maxval=random_shift+1, dtype=tf.int32)
    w_rand = tf.random.uniform(shape=[], minval=-random_shift, maxval=random_shift+1, dtype=tf.int32)

    return [input_bbox[0]+w_rand, input_bbox[1]+h_rand, input_bbox[2]+w_rand, input_bbox[3]+h_rand], h_rand, w_rand


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

def preprocess_pair_test(templar_buffer, search_buffer, templar_mask_buffer, search_mask_buffer, object_id, num_channels, img_h, img_w):
    # decode raw buffers
    templar_img = tf.image.decode_jpeg(templar_buffer, channels=num_channels)  # uint8, [h,w,3]
    search_img = tf.image.decode_jpeg(search_buffer, channels=num_channels)  # uint8, [h,w,3]
    templar_mask = tf.image.decode_png(templar_mask_buffer, channels=1)  # uint8, [h,w,1]
    search_mask = tf.image.decode_png(search_mask_buffer, channels=1)  # uint8, [h,w,1]
    [unique_values, _] = tf.unique(x=tf.reshape(templar_mask, [-1]))
    obj_ids = tf.contrib.framework.sort(values=tf.cast(unique_values, tf.int32)) # tf.int32
    print_op = tf.print('obj id: ', tf.math.reduce_max(templar_mask), 'object ids: ',obj_ids, output_stream=sys.stdout)

    with tf.control_dependencies([print_op]):
        templar_mask = tf.cast(tf.math.equal(templar_mask, tf.cast(obj_ids[object_id], tf.uint8)), tf.uint8)
        search_mask = tf.cast(tf.math.equal(search_mask, tf.cast(obj_ids[object_id], tf.uint8)), tf.uint8)

    object_id = tf.cast(object_id, tf.uint8)
    score = np.array([1,2,3], dtype=np.int32)
    score_weight = np.array([1,2,3], dtype=np.int32)
    gt_masks = np.array([1,2,3], dtype=np.int32)
    tight_temp_bbox = tf.cast(templar_mask, tf.float32)
    tight_search_bbox = tf.cast(search_mask, tf.float32)


    return templar_img, search_img, score, score_weight, gt_masks, tight_temp_bbox, tight_search_bbox

def preprocess_pair(templar_img, search_img, templar_mask, search_mask, object_id, num_channels, img_h, img_w):
    """Preprocesses the give templar/search image buffers and the corresponding masks.
    Args:
    templar_img: decoded JPEG image
    search_img: decoded JPEG image
    templar_mask: decoded PNG image
    search_mask: decoded PNG image
    object_id: the object id of the mask
    num_channels: Integer depth of the image buffer for decoding.

    Returns:
    Pre-processed:
        templar_img + mask: [127, 127, 4]
        search_img + gaussian_mask: [255, 255, 4]
        score: [17, 17, 1]
        score_weight: [17, 17, 1]
        gt_masks: [127, 127, 13], all masks on the positive locations
        tight_temp_bbox: list [xmin, ymin, xmax, ymax], int, only for visualization
        tight_search_bbox: list [xmin, ymin, xmax, ymax], int, only for visualization
    """

    object_id = tf.cast(object_id, tf.uint8)
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(templar_img)[0], tf.cast(img_h, tf.int32)),
                                  tf.debugging.assert_equal(tf.shape(templar_img)[1], tf.cast(img_w, tf.int32))]):
        # extract binary object mask given object_id
        templar_mask = tf.cast(tf.math.equal(templar_mask, object_id), tf.uint8)
        search_mask = tf.cast(tf.math.equal(search_mask, object_id), tf.uint8)

    ######################################## Process Templar #############################################
    # get mean rgb in case of padding
    mean_rgb = tf.reduce_mean(tf.cast(templar_img, tf.int64))
    mean_rgb = tf.cast(mean_rgb, tf.uint8) # tf.uint8
    templar_bbox = bbox_from_mask(templar_mask) # [xmin, ymin, xmax, ymax]
    bbox_h = templar_bbox[3] - templar_bbox[1]
    bbox_w = templar_bbox[2] - templar_bbox[0]
    p = tf.cast((bbox_h + bbox_w) / 4, tf.int32)  # get context margin and compute new bbox
    # save tight bbox for vis
    tight_bbox = []
    tight_bbox.append(templar_bbox[0])
    tight_bbox.append(templar_bbox[1])
    tight_bbox.append(templar_bbox[2])
    tight_bbox.append(templar_bbox[3])
    argmin_dim = tf.math.argmin([bbox_w, bbox_h], axis=0) # 0: shorter in width, 1: shorter in height
    extend_w_cond = tf.equal(argmin_dim, 0) # true if extend in width dim, otherwise extend in height dim
    extend_side_cond = tf.equal(tf.math.abs(bbox_w-bbox_h) % 2, 0) # if true, extend evenly on both side
    extend_val_left = tf.cond(extend_side_cond,
                              lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32),
                              lambda: tf.cast(tf.math.abs(bbox_w - bbox_h) / 2, tf.int32) + 1)
    extend_val_right = tf.cast(tf.math.abs(bbox_w-bbox_h) / 2, tf.int32)
    # get a rect bbox bydistort_bounding_box extending the shorter side
    templar_bbox_new = tf.cond(extend_w_cond, lambda: extend_bbox_w(templar_bbox, extend_val_left, extend_val_right),
                               lambda: extend_bbox_h(templar_bbox, extend_val_left, extend_val_right))
    ## add context margin to bbox
    templar_bbox_new = [templar_bbox_new[0]-p, templar_bbox_new[1]-p, templar_bbox_new[2]+p, templar_bbox_new[3]+p]
    tight_bbox[0] = tight_bbox[0] - templar_bbox_new[0]
    tight_bbox[1] = tight_bbox[1] - templar_bbox_new[1]
    tight_bbox[2] = tight_bbox[2] - templar_bbox_new[0]
    tight_bbox[3] = tight_bbox[3] - templar_bbox_new[1]
    # now the rectangular bbox might be already out of boundary, must pad precise number of pixels on left/up
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
    # pad templar mask with zeros
    templar_mask = tf.pad(tensor=templar_mask,
                          paddings=[[pad_h_begin, pad_h_end + 10], [pad_w_begin, pad_w_end + 10], [0, 0]],
                          mode='CONSTANT', name=None, constant_values=0)
    # crop the image
    croped_templar = tf.image.crop_to_bounding_box(image=templar_img, offset_height=new_y_min,
                                                   offset_width=new_x_min,
                                                   target_height=templar_bbox_new[3] - templar_bbox_new[1],
                                                   target_width=templar_bbox_new[2] - templar_bbox_new[0])
    # crop the mask
    croped_templar_mask = tf.image.crop_to_bounding_box(image=templar_mask, offset_height=new_y_min,
                                                        offset_width=new_x_min,
                                                        target_height=templar_bbox_new[3] - templar_bbox_new[1],
                                                        target_width=templar_bbox_new[2] - templar_bbox_new[0])
    with tf.control_dependencies([tf.debugging.assert_equal(templar_bbox_new[3] - templar_bbox_new[1],
                                                            templar_bbox_new[2] - templar_bbox_new[0])]):
        # rescale to [127, 127], get the scale factor
        scale_s = 127.0 / tf.cast(templar_bbox_new[3] - templar_bbox_new[1], tf.float32)
        # rescale the tight bbox
        tight_temp_bbox = rescale_bbox(tight_bbox, scale_s)
        scale_s = tf.debugging.assert_all_finite(t=scale_s, msg='scale factor not a number!')
        croped_templar = tf.image.resize_bilinear(images=tf.expand_dims(croped_templar, axis=0), size=[127, 127],
                                                  name='resize_templar_img') # tf.float32
        croped_templar = tf.squeeze(croped_templar, axis=0) # [127, 127, 3], tf.float32
        croped_templar_mask = tf.image.resize_bilinear(images=tf.expand_dims(croped_templar_mask, axis=0),
                                                       size=[127, 127], name='resize_templar_mask') # tf.float32
        croped_templar_mask = tf.squeeze(croped_templar_mask, axis=0) # [127, 127, 1], tf.float32
    # check size
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(croped_templar)[0], 127),
                                  tf.debugging.assert_equal(tf.shape(croped_templar)[1], 127),
                                  tf.debugging.assert_equal(tf.shape(croped_templar)[2], 3)]):
        templar_final = tf.identity(croped_templar)
        templar_mask_final = tf.identity(croped_templar_mask)
    templar_img_mask = tf.concat(values=[templar_final, templar_mask_final], axis=-1) # [127, 127, 4], tf.float32

    ######################################## Process Search image #############################################
    # get rgb mean, search_box
    mean_rgb = tf.reduce_mean(tf.cast(search_img, tf.int64))  # tf.uint8
    mean_rgb = tf.cast(mean_rgb, tf.float32)
    search_bbox = bbox_from_mask(search_mask)  # [xmin, ymin, xmax, ymax]
    # get random scale factor
    rescale_factor = scale_s * tf.random.uniform(shape=[], minval=0.8, maxval=1.2, dtype=tf.float32)
    # Get rescaled bbox position, and the image/mask
    search_bbox = rescale_bbox(search_bbox, rescale_factor)
    new_height = tf.cast(tf.cast(tf.shape(search_img)[0], tf.float32) * rescale_factor, tf.int32)
    new_width = tf.cast(tf.cast(tf.shape(search_img)[1], tf.float32) * rescale_factor, tf.int32)
    search_img = tf.image.resize_bilinear(images=tf.expand_dims(search_img, axis=0), size=[new_height, new_width],
                                          name='resize_search_img')
    search_img = tf.squeeze(search_img, axis=0)  # [h, w, 3], tf.float32
    search_mask = tf.image.resize_bilinear(images=tf.expand_dims(search_mask, axis=0), size=[new_height, new_width],
                                           name='resize_search_mask')
    search_mask = tf.squeeze(search_mask, axis=0) # [h, w, 1], tf.float32
    ### randomly shift bbox +-32 pixels, get the shift values and new bbox center
    search_bbox, h_shift, w_shift = distort_bounding_box(input_bbox=search_bbox,
                                                         random_shift=32)  # new box [xmin, ymin, xmax, ymax], h_shift, w_shift
    ### crop around the center of the bbox to [255, 255], if out of boundary, pad with mean rgb value
    img_width = tf.shape(search_img)[1]
    img_height = tf.shape(search_img)[0]
    x_center = tf.cast((search_bbox[2] - search_bbox[0]) / 2, tf.int32) + search_bbox[0] # the shifted center
    y_center = tf.cast((search_bbox[3] - search_bbox[1]) / 2, tf.int32) + search_bbox[1] # the shifted center
    x_min, x_max = x_center - 127, x_center + 127
    y_min, y_max = y_center - 127, y_center + 127
    [new_x_min, pad_w_begin] = tf.cond(x_min < 0, lambda: return_zero_pad(x_min), lambda: return_iden_no_pad(x_min))
    [new_x_max, pad_w_end] = tf.cond(x_max >= img_width, lambda: return_maxW_pad(x_max, img_width),
                                     lambda: return_iden_no_pad(x_max))
    [new_y_min, pad_h_begin] = tf.cond(y_min < 0, lambda: return_zero_pad(y_min), lambda: return_iden_no_pad(y_min))
    [new_y_max, pad_h_end] = tf.cond(y_max >= img_height, lambda: return_maxH_pad(y_max, img_height),
                                     lambda: return_iden_no_pad(y_max))
    # do paddings, only effective if out of boundary
    search_img = search_img - mean_rgb
    search_img = tf.pad(tensor=search_img,
                        paddings=[[pad_h_begin, pad_h_end + 10], [pad_w_begin, pad_w_end + 10], [0, 0]],
                        mode='CONSTANT', name=None, constant_values=0)
    search_img = search_img + mean_rgb
    search_mask = tf.pad(tensor=search_mask,
                         paddings=[[pad_h_begin, pad_h_end + 10], [pad_w_begin, pad_w_end + 10], [0, 0]],
                         mode='CONSTANT', name=None, constant_values=0)
    # crop
    search_final = tf.image.crop_to_bounding_box(image=search_img, offset_height=new_y_min, offset_width=new_x_min,
                                                 target_height=255, target_width=255) # [255, 255, 3], tf.float32
    search_mask = tf.image.crop_to_bounding_box(image=search_mask, offset_height=new_y_min, offset_width=new_x_min,
                                                 target_height=255, target_width=255) # [255, 255, 1], tf.float32
    ## get tight bbox within the rescaled search img [xmin, ymin, xmax, ymax]
    bbox_h_half = tf.cast((search_bbox[3] - search_bbox[1]) / 2, tf.int32)  # might be zero
    bbox_w_half = tf.cast((search_bbox[2] - search_bbox[0]) / 2, tf.int32)  # might be zero
    tight_search_bbox = []
    tight_search_bbox.append(127 - bbox_w_half - w_shift)  # xmin
    tight_search_bbox.append(127 - bbox_h_half - h_shift)  # ymin
    tight_search_bbox.append(127 + bbox_w_half - w_shift)  # xmax
    tight_search_bbox.append(127 + bbox_h_half - h_shift)  # ymax
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(search_final)[0], 255),
                                  tf.debugging.assert_equal(tf.shape(search_final)[1], 255),
                                  tf.debugging.assert_equal(tf.shape(search_final)[2], 3)]):
        search_final = tf.identity(search_final)
        search_final_mask = tf.identity(search_mask)
    # get 2d cosine window
    cos_window = np.dot(np.expand_dims(np.hanning(255), 1),
                          np.expand_dims(np.hanning(255), 0))
    cos_window = cos_window / np.sum(cos_window)  # normalize window, [255, 255]
    cos_window = np.expand_dims(cos_window, axis=-1) # [255, 255, 1]
    search_img_mask = tf.concat(values=[search_final, cos_window], axis=-1)  # [255, 255, 4], tf.float32

    ######################################## Process Score Map GT #############################################
    t_center_x = 8 - tf.cast(w_shift / 8, tf.int32)
    t_center_y = 8 - tf.cast(h_shift / 8, tf.int32)
    score, score_weight = tf.py_func(func=build_gt_py, inp=[t_center_x, t_center_y], Tout=[tf.int32, tf.float32],
                                     stateful=True, name=None)
    # check size
    with tf.control_dependencies([tf.debugging.assert_equal(tf.shape(score)[0], 17),
                                  tf.debugging.assert_equal(tf.shape(score)[1], 17),
                                  tf.debugging.assert_equal(tf.shape(score)[2], 1),
                                  tf.debugging.assert_equal(tf.shape(score_weight)[0], 17),
                                  tf.debugging.assert_equal(tf.shape(score_weight)[1], 17),
                                  tf.debugging.assert_equal(tf.shape(score_weight)[2], 1)]):
        score = tf.identity(score) # [17, 17, 1]
        score_weight = tf.identity(score_weight) # [17, 17, 1]

    ################################### Randomly flip templar/search images ####################################
    # flip_v = tf.random.uniform(shape=[]) # scalar
    # flip_v = tf.greater_equal(flip_v, 0.5)
    flip_v = tf.greater(x=1, y=2)
    templar_img_mask = tf.cond(flip_v, lambda : tf.image.flip_left_right(image=templar_img_mask), lambda :templar_img_mask)
    search_img_mask = tf.cond(flip_v, lambda: tf.image.flip_left_right(image=search_img_mask), lambda: search_img_mask)
    score = tf.cond(flip_v, lambda :tf.image.flip_left_right(image=score), lambda :score)
    score_weight = tf.cond(flip_v, lambda :tf.image.flip_left_right(image=score_weight), lambda :score_weight)
    tight_search_bbox = tf.cond(flip_v, lambda :flip_bbox(tight_search_bbox, 255), lambda :tight_search_bbox)
    # gt_masks = tf.cond(flip_v, lambda :tf.image.flip_left_right(image=gt_masks), lambda :gt_masks)
    search_final_mask = tf.cond(flip_v, lambda :tf.image.flip_left_right(image=search_final_mask), lambda :search_final_mask)

    templar_img_mask = mean_image_subtraction(templar_img_mask, _CHANNEL_MEANS, num_channels)
    search_img_mask = mean_image_subtraction(search_img_mask, _CHANNEL_MEANS, num_channels)
    search_img_mask = tf.concat([search_img_mask, search_final_mask], axis=-1) # [255, 255, 5], tf.float32

    ######################################## Process Mask GT ###############################################
    # search_final_mask: [255, 255, 1], tf.float32, already flipped
    tmp_bbox = bbox_from_mask(search_final_mask)  # [xmin, ymin, xmax, ymax]
    x_center = tf.cast((tmp_bbox[2] - tmp_bbox[0]) / 2, tf.int32) + tmp_bbox[0]
    y_center = tf.cast((tmp_bbox[3] - tmp_bbox[1]) / 2, tf.int32) + tmp_bbox[1]
    # crop around [x_center, y_center] with a shift to [127, 127], the shift is pre-defined according to positive locations
    # the order is from left to right, top to down
    shift_vec = [[-16, 0], [-8, -8], [-8, 0], [-8, 8], [0, -16], [0, -8], [0, 0], [0, 8], [0, 16],
                 [8, -8], [8, 0], [8, 8], [16, 0]]
    gt_mask_list = []
    gt_mask_weight_list = []
    for i in range(13):
        tmp_mask = get_mask(center_vec=[x_center, y_center], shift_vec=shift_vec[i],
                            mask=search_final_mask)  # [127, 127, 1] mask, tf.int32
        gt_mask_list.append(tmp_mask)
        tmp_weight = get_mask_weight(tmp_mask)
        gt_mask_weight_list.append(tmp_weight)
    gt_masks = tf.concat(gt_mask_list, axis=-1)  # [127, 127, 13], tf.int32
    gt_masks_weight = tf.concat(gt_mask_weight_list, axis=-1) # [127, 127, 13], tf.float32

    return templar_img_mask, search_img_mask, score, score_weight, gt_masks, tight_temp_bbox, tight_search_bbox, gt_masks_weight

def get_mask(center_vec, shift_vec, mask):
    '''
    :param center_vec: list: [x_center, y_center] of the mask
    :param shift_vec: [shift_h, shift_w]
    :param mask: [255, 255, 1], the original input mask, tf.float32
    :return: [127, 127, 1] mask, tf.int32
    '''

    # get the 4-corners of the mask bbox
    x_min, x_max = center_vec[0] + shift_vec[1] - 63, center_vec[0] + shift_vec[1] + 63
    y_min, y_max = center_vec[1] + shift_vec[0] - 63, center_vec[1] + shift_vec[0] + 63
    # now the bbox might be out of boundary, compute padding values
    [new_x_min, pad_w_begin] = tf.cond(x_min < 0, lambda: return_zero_pad(x_min), lambda: return_iden_no_pad(x_min))
    [new_x_max, pad_w_end] = tf.cond(x_max >= 255, lambda: return_maxW_pad(x_max, 255),
                                     lambda: return_iden_no_pad(x_max))
    [new_y_min, pad_h_begin] = tf.cond(y_min < 0, lambda: return_zero_pad(y_min), lambda: return_iden_no_pad(y_min))
    [new_y_max, pad_h_end] = tf.cond(y_max >= 255, lambda: return_maxH_pad(y_max, 255),
                                     lambda: return_iden_no_pad(y_max))
    # do padding
    pad_mask = tf.pad(tensor=mask,
                      paddings=[[pad_h_begin, pad_h_end + 1], [pad_w_begin, pad_w_end + 1], [0, 0]],
                      mode='CONSTANT', name=None, constant_values=0)
    # crop to bbox
    crop_mask = tf.image.crop_to_bounding_box(image=pad_mask, offset_height=new_y_min, offset_width=new_x_min,
                                              target_height=127, target_width=127)  # [127, 127, 1], tf.float32
    # binarize mask
    bin_mask = tf.cast(tf.math.greater_equal(crop_mask, 0.5), tf.int32) # [127, 127, 1], tf.int32

    return bin_mask

def get_mask_weight(mask):
    '''
    :param mask: [127, 127, 1], binary mask, tf.int32
    :return: [127, 127, 1], tf.float32
    '''

    num_total = 127 * 127
    num_positive = tf.math.reduce_sum(mask)
    num_negative = num_total - num_positive
    weight_positive = tf.cast(num_negative, tf.float32) / tf.cast(num_total, tf.float32)
    weight_negative = tf.cast(num_positive, tf.float32) / tf.cast(num_total, tf.float32)
    mat_positive = tf.cast(mask, tf.float32) * weight_positive  # [127, 127, 1] tf.float32
    mat_negative = (1.0 - tf.cast(mask, tf.float32)) * weight_negative  # [127, 127, 1] tf.float32
    score_weight = mat_positive + mat_negative

    return score_weight


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

def mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.
    Args:
    image: a tensor of size [height, width, 4].
    means: a 3-vector of values to subtract from RGB channels.
    num_channels: number of color channels in the image that will be distorted.
    Returns:
    the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')

    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    # extract RGB channels
    rgb_img = image[:, :, 0:3] # [h, w, 3]
    means = tf.expand_dims(tf.expand_dims(means, 0), 0) # [1,1,3]
    new_img = rgb_img - means # [h, w, 3]

    # assemble the mask channel
    new_img = tf.concat([new_img, image[:,:,3:4]], axis=-1) # [h, w, 4]

    return new_img

def filter_bbox(parsed_dict):
    '''
    :param parsed_dict:
    :return: true if examples hold the condition
    '''
    # filter the examples whose tight bbox height or width <8
    # [xmin, ymin, xmax, ymax]

    temp_tight_bbox = parsed_dict['tight_temp_bbox']
    search_tight_bbox = parsed_dict['tight_search_bbox']
    keep_bool0 = tf.logical_and(tf.greater_equal(temp_tight_bbox[2] - temp_tight_bbox[0], 8),
                                tf.greater_equal(temp_tight_bbox[3] - temp_tight_bbox[1], 8))
    keep_bool1 = tf.logical_and(tf.greater_equal(search_tight_bbox[2] - search_tight_bbox[0], 8),
                                tf.greater_equal(search_tight_bbox[3] - search_tight_bbox[1], 8))
    keep_bool = tf.logical_and(keep_bool0, keep_bool1)

    return keep_bool

def filter_mask(parsed_dict):
    '''
    :param parsed_dict:
    :return: true if examples hold the condition
    '''
    # filter the examples whose:
    #   * templar mask size < 16x16
    #   * gt_masks size < 16x16

    temp_mask = parsed_dict['templar_img_mask'][:,:,3:4] # [127, 127, 1], float mask
    temp_mask = tf.cast(tf.math.greater_equal(temp_mask, 0.5), tf.int32)  # [127, 127, 1], binary mask, tf.int32
    gt_masks = parsed_dict['gt_masks'] # [127, 127, 13], binary mask, tf.int32

    bool0 = tf.greater_equal(tf.math.reduce_sum(temp_mask), 16*16)
    bool1 = tf.greater_equal(tf.math.reduce_sum(gt_masks[:,:,6:7]), 16*16) # only check the central mask size
    keep_bool = tf.logical_and(bool0, bool1)

    return keep_bool

def reformat_channel_first(example_dict):

    templar_img_mask = tf.transpose(example_dict['templar_img_mask'], [2, 0, 1]) # from [h, w, c] to [c, h, w]
    search_img_mask = tf.transpose(example_dict['search_img_mask'], [2, 0, 1])
    score = tf.transpose(example_dict['score'], [2, 0, 1])
    score_weight = tf.transpose(example_dict['score_weight'], [2, 0, 1])
    gt_masks = tf.transpose(example_dict['gt_masks'], [2, 0, 1])
    gt_masks_weight = tf.transpose(example_dict['gt_masks_weight'], [2, 0, 1])

    dict = {'templar_img_mask':templar_img_mask, 'search_img_mask':search_img_mask, 'score':score,
            'score_weight':score_weight, 'gt_masks': gt_masks, 'tight_temp_bbox': example_dict['tight_temp_bbox'],
            'tight_search_bbox': example_dict['tight_search_bbox'], 'gt_masks_weight': gt_masks_weight}

    return dict


########################################################################
# Build TF Dataset of ImageNet15-VID train pipeline
########################################################################
def build_dataset(num_gpu=2, batch_size=8, train_record_dir='/storage/slurm/wangyu/youtube_vos/tfrecord_train',
                  data_format='channels_first'):
    """
    Note that though each gpu process a unique part of the dataset, the data pipeline is built
    only on CPU. Each GPU worker will have its own dataset iterator

    :param num_gpu: total number of gpus, each gpu will be assigned to a unique part of the dataset
    :param batch_size: batch of images processed on each gpu
    :param train_record_dir: where the tfrecord files are stored
    :param data_format: must be 'channels_first' when using gpu
    :return: a list of sub-dataset for each different gpu
    """


    file_list = get_filenames(train_record_dir)
    print('Got {} tfrecords.'.format(len(file_list)))
    subset = []

    # create dataset, reading all train tfrecord files, default number of files = _NUM_SHARDS
    dataset = tf.data.Dataset.list_files(file_pattern=file_list, shuffle=False)
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
                                                   cycle_length=10, block_length=2, num_parallel_calls=4)
        subset[gpu_id] = subset[gpu_id].prefetch(buffer_size=batch_size*2) # prefetch
        subset[gpu_id] = subset[gpu_id].map(parse_raw, num_parallel_calls=4) # parallel parse 4 examples at once
        subset[gpu_id] = subset[gpu_id].filter(filter_empty) # filter empty masks
        subset[gpu_id] = subset[gpu_id].map(parse_func, num_parallel_calls=4) # parallel parse 4 examples at once
        subset[gpu_id] = subset[gpu_id].filter(filter_bbox)  # filter tiny bbox
        subset[gpu_id] = subset[gpu_id].filter(filter_mask) # filter tiny masks
        if data_format == 'channels_first':
            subset[gpu_id] = subset[gpu_id].map(reformat_channel_first, num_parallel_calls=4) # parallel parse 4 examples at once
        else:
            raise ValueError('Data format is not channels_first when building dataset pipeline!')
        subset[gpu_id] = subset[gpu_id].shuffle(buffer_size=100)
        subset[gpu_id] = subset[gpu_id].repeat()
        subset[gpu_id] = subset[gpu_id].batch(batch_size) # inference batch images for one feed-forward
        # prefetch and buffer internally, to prevent starvation of GPUs
        subset[gpu_id] = subset[gpu_id].prefetch(buffer_size=batch_size*2)

    print('Dataset pipeline built for {} GPUs.'.format(num_gpu))

    return subset