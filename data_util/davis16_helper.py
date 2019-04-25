'''
    Helper functions for davis16 tracking and segmentation
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from PIL import Image
import numpy as np

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
