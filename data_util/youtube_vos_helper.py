from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw


def get_masks(gt_file):
    '''
    :param gt_file: any *.png file that encodes object ids
    :return: a list of binary masks, [[obj_id0, mask0], [obj_id1, mask1], ...] as hxwx1 image
    '''

    bin_masks = []
    seg_obj = Image.open(gt_file)
    seg_arr = np.array(seg_obj)
    obj_ids = np.unique(seg_arr)  # obj ids: [0, 1, 2, ...]
    if 0 not in obj_ids:
        raise ValueError('0 not in gt_file {}'.format(gt_file))
    obj_ids = obj_ids[1:]
    for obj_id in obj_ids:
        mask = (seg_arr == obj_id)
        mask = mask.astype(np.uint8)
        bin_masks.append([obj_id, np.expand_dims(mask, -1)])

    return bin_masks


def bbox_from_mask(masks):
    '''
    :param masks: [[id0, mask0], [id1, mask1], ...] where masks are binary hxwx1 np.array
    :return: [bbox0, bbox1, ...] as [xmin, ymin, xmax, ymax]
    '''

    bbox_list = []
    for item in masks:
        mask = np.squeeze(item[1], -1)  # hxw binary mask
        indices = np.nonzero(mask)  # (x_indices, y_indices)
        min_x = indices[1].min()
        max_x = indices[1].max() + 1  # as exclusive
        min_y = indices[0].min()
        max_y = indices[0].max() + 1  # as exclusive
        bbox_list.append([min_x, min_y, max_x, max_y])

    return bbox_list


def open_img(img_path):
    img_obj = Image.open(img_path)
    img_arr = np.array(img_obj)

    return img_obj, img_arr


def draw_bbox(img_obj, bbox, save_dir, gt_bool=False):
    '''
    :param img_obj: PIL Image Object
    :param bbox: [xmin, ymin, xmax, ymax]
    :return:
    '''
    if gt_bool:
        color = (200, 10, 10)
    else:
        color = (10, 200, 10)

    draw_handle = Draw(img_obj)
    draw_handle.rectangle(bbox, outline=color)

    img_obj.save(save_dir)
    print('Draw bbox {}'.format(save_dir))


def draw_bbox_mask(img_arr, mask_arr, bbox, save_dir, gt_bool=False):
    '''
    :param img_arr: [h, w, 3]
    :param mask_arr: [h, w, 1], uint8
    :param bbox: [xmin, ymin, xmax, ymax]
    :param save_dir:
    :param gt_bool:
    :return:
    '''

    # create redish mask
    mask_r = mask_arr * 96
    mask_g = mask_arr * 0
    mask_b = mask_arr * 0
    if mask_arr.max() != 1:
        raise ValueError('max != 1')
    mask_rgb = np.concatenate([mask_r, mask_g, mask_b], axis=-1)  # [h,w,3], uint8
    blended = img_arr.astype(np.int32) + mask_rgb.astype(np.int32)
    blended = np.clip(blended, 0, 255)
    img_obj = Image.fromarray(blended.astype(np.uint8))

    # draw bbox
    if gt_bool:
        color = (200, 10, 10)
    else:
        color = (10, 200, 10)

    draw_handle = Draw(img_obj)
    draw_handle.rectangle(bbox, outline=color)

    img_obj.save(save_dir)
    print('Draw bbox {}'.format(save_dir))
