'''
    Helper functions for davis16 tracking and segmentation
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from PIL import Image
import numpy as np
from PIL.ImageDraw import Draw

sys.path.append('..')

result_base = '/storage/slurm/wangyu/davis16'
results_parts_assemble = os.path.join(result_base, 'results_parts_assemble')
results_parts_assemble_color = os.path.join(result_base, 'results_parts_assemble_color')

val_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
            'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
            'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']
images_base = '/usr/stud/wangyu/DAVIS17_train_val/JPEGImages/480p'


def assembel_parts_prob(parts_list, seq_name, frame_id, th=0.55, save_file=True):
    '''
    :param parts_list: [prob_arr0, prob_arr1, ...] where prob_arrx is [480, 854], probability np.float32
    :param seq_name: must be one of the names in val_list
    :param frame_id:
    :param th: threshold for binary mask, default 0.55
    :param save_file: default is True
    :return: [h,w] np.float32 as probability map (soft attention)
             [h,w] np.uint8 as binary mask (hard attention)
    '''

    num_parts = len(parts_list)
    acc_sum = np.zeros_like(parts_list[0], dtype=np.float32)  # [480, 854], zeros, np.float32
    acc_prob = np.zeros_like(parts_list[0], dtype=np.float32)  # [480, 854], zeros, np.float32
    for i in range(num_parts):
        # acc prob map
        prob_arr = parts_list[i]
        acc_prob += prob_arr

        # acc non-zero
        bin_arr = prob_arr
        bin_arr[bin_arr > 0.01] = 1.0
        acc_sum += bin_arr

    # set zeros to ones for acc_sum
    acc_sum[acc_sum < 0.1] = 1.0
    final_prob = np.divide(acc_prob, acc_sum)  # [480, 854] final prob_map
    bin_mask = final_prob
    bin_mask[bin_mask < th] = 0
    bin_mask[bin_mask >= th] = 1
    bin_mask = bin_mask.astype(np.uint8)

    if save_file:
        # save binary mask to file
        save_mask_path = os.path.join(results_parts_assemble, seq_name, str(frame_id).zfill(5) + '.png')
        bin_obj = Image.fromarray(bin_mask)
        bin_obj.putpalette([0, 0, 0, 192, 32, 32])
        bin_obj.save(save_mask_path)

        # save colored rgb to file
        save_color_path = os.path.join(results_parts_assemble_color, seq_name, str(frame_id).zfill(5) + '.jpg')
        rgb_obj = Image.open(os.path.join(images_base, seq_name, str(frame_id).zfill(5) + '.jpg'))
        blended = blend_rgb_mask(img_arr=np.array(rgb_obj),
                                 mask=np.expand_dims(bin_mask, -1))
        blended.save(save_color_path)

    return final_prob, bin_mask


def assembel_bbox(img_arr, boxes, sequence_name, frame_id):
    save_mask_path = os.path.join(results_parts_assemble, sequence_name, str(frame_id).zfill(5) + '.png')
    save_color_path = os.path.join(results_parts_assemble_color, sequence_name, str(frame_id).zfill(5) + '.jpg')

    num_bbox = len(boxes)
    assembled_mask = np.zeros((480, 854), np.int32)
    for i in range(num_bbox):
        box = boxes[i]
        zeros_arr = np.zeros((480, 854), np.uint8)  # (480, 854)
        zeros_obj = Image.fromarray(zeros_arr)  # [854, 480]
        # draw and fill
        draw_handle = Draw(zeros_obj)
        draw_handle.rectangle(xy=box, fill=1, outline=0)
        part_mask = np.array(zeros_obj, np.int32)
        assembled_mask += part_mask

    final_mask = assembled_mask
    final_mask[final_mask >= 1] = 1  # as the final attention mask
    final_mask = final_mask.astype(np.uint8)
    bin_obj = Image.fromarray(final_mask)
    bin_obj.putpalette([0, 0, 0, 192, 32, 32])
    bin_obj.save(save_mask_path)

    blended = blend_rgb_mask(img_arr=img_arr, mask=np.expand_dims(final_mask, -1))
    blended.save(save_color_path)

def blend_rgb_mask(img_arr, mask):
    '''
    :param img_arr: [480, 854, 3]
    :param mask: [480, 854, 1]
    :return: Image object
    '''
    # create redish mask
    mask_r = mask * 192
    mask_g = mask * 32
    mask_b = mask * 32
    mask_rgb = np.concatenate([mask_r, mask_g, mask_b], axis=-1)
    blended = img_arr.astype(np.int32) + mask_rgb.astype(np.int32)
    blended = np.clip(blended, 0, 255)
    img_obj = Image.fromarray(blended.astype(np.uint8))

    return img_obj


def get_valid_indices(valid_path):
    '''
    :param valid_path: /storage/slurm/wangyu/davis16/valid_indices/blackswan/00001.npy
    :param frame_id: scalar from [1,2,3,...]
    :return: a list of arbitrary length
    '''

    valid_arr = np.load(valid_path)
    valid_indices = valid_arr.tolist()

    return valid_indices


def get_pre_bbox(pre_bbox_path):
    '''
    :param pre_bbox_path: /storage/slurm/wangyu/davis16/pre_bboxes/blackswan/00001.npy
    :param frame_id: scalar from [1,2,3,...]
    :return: a list of bboxes: [bbox0, bbox1, ...], number of bboxes = number of init_bboxes
    '''

    bbox_arr = np.load(pre_bbox_path)  # (n,4)
    num_box = np.shape(bbox_arr)[0]
    if np.shape(bbox_arr)[1] != 4:
        raise ValueError('bad loaded bbox array shape: {}'.format(np.shape(bbox_arr)))
    # convert to list
    bbox_list = []
    for i in range(num_box):
        bbox_list.append(bbox_arr[i].tolist())

    return bbox_list


def filter_bbox_mask(prob_mask, bin_mask, tracked_box):
    '''
    :param prob_mask: [480, 854], np.float32
    :param bin_mask: [480, 854], np.uint8
    :param tracked_box: [xmin, ymin, xmax, ymax]
    :return:
    '''

    zeros_arr = np.zeros_like(bin_mask, np.uint8)  # (480, 854)
    zeros_obj = Image.fromarray(zeros_arr)  # [854, 480]
    # draw and fill
    draw_handle = Draw(zeros_obj)
    draw_handle.rectangle(xy=tracked_box, fill=1, outline=1)
    gate_mask = np.array(zeros_obj, np.float32)

    # new prob_mask
    prob_mask = np.multiply(prob_mask, gate_mask)
    bin_mask = np.multiply(bin_mask, gate_mask.astype(np.uint8))

    return prob_mask, bin_mask
