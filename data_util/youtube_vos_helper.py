from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw
from random import shuffle


def get_masks(gt_file_list):
    '''
    :param gt_file_list: a list of part mask files
    :return: a list of binary masks, [[obj_id0, mask0], [obj_id1, mask1], ...] as hxwx1 image
    '''

    bin_masks = []
    num_parts = len(gt_file_list)
    for i in range(num_parts):
        gt_file = gt_file_list[i]
        seg_obj = Image.open(gt_file)  # (854, 480)
        seg_arr = np.array(seg_obj)  # [480, 854]
        obj_ids = np.unique(seg_arr)  # obj ids: [0, 1]
        if (0 not in obj_ids) or (1 not in obj_ids):
            raise ValueError('0 or 1 not in gt_file {}'.format(gt_file))
        if obj_ids.max() != 1:
            raise ValueError('obj_id max != 1 for gt_file {}'.format(gt_file))
        seg_arr = seg_arr.astype(np.uint8)
        bin_masks.append([i, np.expand_dims(seg_arr, -1)])

    return bin_masks


def bbox_from_mask(masks):
    '''
    :param masks: [[id0, mask0], [id1, mask1], ...] where masks are binary hxwx1 np.array
    :return: [bbox0, bbox1, ...] as [xmin, ymin, xmax, ymax]
    '''

    bbox_list = []
    for item in masks:
        mask = np.squeeze(item[1], -1)  # hxw binary mask
        indices = np.nonzero(mask)  # (h_indices, w_indices)
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
        print('max != 1')
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
    # print('Draw bbox {}'.format(save_dir))


def draw_mul_bbox_mask(img_arr, mask_arrs, boxes, colors, indices, save_dir):
    '''
    :param img_arr: [h, w, 3], np.array
    :param mask_arrs: list of mask_arr [[obj_id0, mask0], [obj_id1, mask1], ...] as hxwx1
    :param boxes: list of boxes, each is [xmin, ymin, xmax, ymax]
    :param colors: list of possible colors
    :param indices: list of part indices to be drawn
    :param save_dir:
    :return: None
    '''

    num_bbox = len(boxes)
    # if len(mask_arrs) != len(boxes):
    #    raise ValueError('Num of masks {} != num of boxes {}'.format(len(mask_arrs), len(boxes)))

    # blend masks
    blended = img_arr.astype(np.int32)
    for i in range(num_bbox):
        #mask_arr = mask_arrs[i][1]

        # create redish mask
        # mask_r = mask_arr * 96
        # mask_g = mask_arr * 0
        # mask_b = mask_arr * 0
        #mask_rgb = np.concatenate([mask_r, mask_g, mask_b], axis=-1)  # [h,w,3], uint8
        # blended += mask_rgb.astype(np.int32)
        blended = np.clip(blended, 0, 255)

    # draw boxes
    img_obj = Image.fromarray(blended.astype(np.uint8))
    draw_handle = Draw(img_obj)
    for i in range(num_bbox):
        bbox = boxes[i]
        draw_handle.rectangle(bbox, outline=colors[i])

    # save
    img_obj.save(save_dir)


def gen_box_colors():
    # generate maximum 448 colors
    r_colors = [24, 48, 72, 96, 120, 144, 168, 192]  # r_num = 8
    g_colors = [24, 48, 72, 96, 120, 144, 168, 192]  # g_num = 8
    b_colors = [24, 48, 72, 96, 120, 144, 168, 192]  # b_num = 7
    shuffle(r_colors)
    shuffle(g_colors)
    shuffle(b_colors)

    colors = []
    for r_i in r_colors:
        for g_i in g_colors:
            for b_i in b_colors:
                color = (r_i, g_i, b_i)
                colors.append(color)

    return colors
