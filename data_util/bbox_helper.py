'''
    Some of the bbox operations used
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def draw_bbox_templar(templars, bbox_templars, batch):
    '''
    :param templars: [batch, 3, 127, 127]
    :param bbox_templars: [batch, 4], [xmin, ymin, xmax, ymax]
    :param batch: batch size
    :return: drawn templars [batch, 127, 127, 3]
    '''

    # [batch, num_bounding_boxes, 4]
    batch_bbox_list = [] # [[1,4], [1,4], ...]
    for batch_i in range(batch):
        # get tight bbox
        tmp_box = tf.cast(bbox_templars[batch_i], tf.float32)
        tight_bbox = [tmp_box[1] / 127.0, tmp_box[0] / 127.0, tmp_box[3] / 127.0, tmp_box[2] / 127.0] # [y_min, x_min, y_max, x_max]
        batch_i_bbox = tf.stack(values=[tight_bbox], axis=0) # [1, 4]
        batch_bbox_list.append(batch_i_bbox)
    # stack batch
    batch_bbox = tf.stack(values=batch_bbox_list, axis=0) # [batch, 1, 4]
    drawed_images = tf.image.draw_bounding_boxes(images=tf.transpose(templars, [0,2,3,1]),
                                                 boxes=batch_bbox,
                                                 name='draw_temp_bbox')

    return drawed_images

def draw_bbox_search(searchs, bbox_searchs, batch):
    '''
    :param searchs: [batch, 3, 255, 255]
    :param bbox_searchs: [batch, 4], [xmin, ymin, xmax, ymax]
    :param batch:  batch size
    :return: drawn searchs [batch, 255, 255, 3]
    '''

    batch_bbox_list = []  # [[1,4], [1,4], ...]
    for batch_i in range(batch):
        # get tight bbox
        tmp_box = tf.cast(bbox_searchs[batch_i], tf.float32)
        tight_bbox = [tmp_box[1] / 255.0, tmp_box[0] / 255.0, tmp_box[3] / 255.0,
                      tmp_box[2] / 255.0]  # [y_min, x_min, y_max, x_max]
        tight_bbox = tf.expand_dims(tight_bbox, axis=0) # [1, 4]
        batch_bbox_list.append(tight_bbox)
    # stack batch
    batch_bbox = tf.stack(values=batch_bbox_list, axis=0)  # [batch, 1, 4]
    drawed_images = tf.image.draw_bounding_boxes(images=tf.transpose(searchs, [0, 2, 3, 1]),
                                                 boxes=batch_bbox,
                                                 name='draw_search_bbox')

    return drawed_images

def blend_rgb_mask(img_mask, batch):
    '''
    :param img_mask: [n, 4, h, w], tf.float32
    :return: [n, h, w, 3], tf.float32
    '''

    blended_list = []
    for batch_i in range(batch):
        mask = img_mask[batch_i:batch_i+1,3:4,:,:] # [1,1,h,w], tf.float32
        # create redish mask
        mask_r = mask * 128
        mask_g = mask * 32
        mask_b = mask * 64
        mask_rgb = tf.concat([mask_r, mask_g, mask_b], axis=1) # [1,3,h,w], tf.float32
        rgb_img = img_mask[batch_i:batch_i+1,0:3,:,:] # [1,3,h,w], tf.float32
        blended = tf.transpose(rgb_img + mask_rgb, [0, 2, 3, 1]) # [1,3,h,w], tf.float32
        blended_list.append(blended)
    blended = tf.concat(values=blended_list, axis=0) # [n, h, w, 3], tf.float32

    return blended

def blend_search_seg_mask(img_search, gt_masks, centers, batch):
    '''
    :param img_search: [n, 3, 255, 255]
    :param gt_masks: [n, 13, 127, 127]
    :param centers: [n, 2]
    :param batch:
    :return: [n*13, 127, 127, 3]
    '''

    shift_map = [[-16, 0], [-8, -8], [-8, 0], [-8, 8], [0, -16], [0, -8], [0, 0], [0, 8], [0, 16],
                 [8, -8], [8, 0], [8, 8], [16, 0]]

    blended_list = []
    for batch_i in range(batch):
        center_vec = centers[batch_i]
        img = img_search[batch_i:batch_i + 1, :, :, :]  # [1, 3, 255, 255]
        img = tf.transpose(tf.squeeze(img, axis=0), [1, 2, 0])  # [255, 255, 3]
        mean_rgb = tf.reduce_mean(img)
        # for each segment
        for idx_i in range(13):
            # extract mask
            mask = tf.cast(gt_masks[batch_i:batch_i+1, idx_i:idx_i+1, :, :], tf.float32) # [1, 1, 127, 127], tf.float32
            mask = tf.squeeze(mask, axis=0) # [1, 127, 127], tf.float32
            mask_r = mask * 128
            mask_g = mask * 32
            mask_b = mask * 64
            mask_rgb = tf.concat([mask_r, mask_g, mask_b], axis=0)  # [3,127,127], tf.float32
            mask_rgb = tf.transpose(mask_rgb, [1, 2, 0]) # [127,127,3], tf.float32
            # extract rgb image segment
            shift_vec = shift_map[idx_i]
            x_min, x_max = center_vec[0] + shift_vec[1] - 63, center_vec[0] + shift_vec[1] + 63
            y_min, y_max = center_vec[1] + shift_vec[0] - 63, center_vec[1] + shift_vec[0] + 63
            [new_x_min, pad_w_begin] = tf.cond(x_min < 0, lambda: return_zero_pad(x_min),
                                               lambda: return_iden_no_pad(x_min))
            [new_x_max, pad_w_end] = tf.cond(x_max >= 255, lambda: return_maxW_pad(x_max, 255),
                                             lambda: return_iden_no_pad(x_max))
            [new_y_min, pad_h_begin] = tf.cond(y_min < 0, lambda: return_zero_pad(y_min),
                                               lambda: return_iden_no_pad(y_min))
            [new_y_max, pad_h_end] = tf.cond(y_max >= 255, lambda: return_maxH_pad(y_max, 255),
                                             lambda: return_iden_no_pad(y_max))
            img = img - mean_rgb
            img = tf.pad(tensor=img,
                         paddings=[[pad_h_begin, pad_h_end + 1], [pad_w_begin, pad_w_end + 1], [0, 0]],
                         mode='CONSTANT', name=None, constant_values=0)
            img = img + mean_rgb
            crop_img = tf.image.crop_to_bounding_box(image=img, offset_height=new_y_min, offset_width=new_x_min,
                                                     target_height=127, target_width=127)  # [127, 127, 3], tf.float32
            blended = mask_rgb + crop_img # [127, 127, 3]
            blended_list.append(blended)
    blended_stack = tf.stack(values=blended_list, axis=0) # [n*13, 127, 127, 3]

    return blended_stack

def get_mask_center(masks, batch):
    '''
    :param masks: [n, 1, 255, 255], tf.float32
    :return: [n, 2], where the 2nd-dim is [center_x, center_y]
    '''

    # binarize masks
    bin_mask = tf.cast(tf.math.greater_equal(masks, 0.5), tf.uint8) # [n, 1, 255, 255], tf.uint8
    zero_t = tf.constant(0, dtype=tf.uint8)

    center_list = []
    for batch_i in range(batch):
        bool_mat = tf.math.not_equal(x=tf.squeeze(bin_mask[batch_i:batch_i+1,:,:,:], axis=[0,1]), y=zero_t, name=None)  # [255, 255]
        indices = tf.where(bool_mat)  # [n, 2], n is number of non-zeros, the other dim gives the index
        max_v = tf.math.reduce_max(indices, axis=[0], keepdims=False)  # (max_h, max_w)
        min_v = tf.math.reduce_min(indices, axis=[0], keepdims=False)  # (min_h, min_w)
        max_h = tf.cast(max_v[0], tf.int32)
        max_w = tf.cast(max_v[1], tf.int32)
        min_h = tf.cast(min_v[0], tf.int32)
        min_w = tf.cast(min_v[1], tf.int32)
        center_x = tf.cast((max_w-min_w+1)/2, tf.int32) + min_w
        center_y = tf.cast((max_h-min_h+1)/2, tf.int32) + min_h
        center_list.append([center_x, center_y])

    return center_list

def return_zero_pad(x): return [0, tf.abs(x)]
def return_iden_no_pad(x): return [x, 0]
def return_maxW_pad(x, w_max): return [w_max - 1, x - (w_max - 1)]
def return_maxH_pad(x, h_max): return [h_max - 1, x - (h_max - 1)]