'''
This script generates a pairs of images/masks simulating two frames from the same video, given a single image/mask.
The given image may have arbitrary number of objects labeled as [1, 2, ... M]
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2, sys, os
from PIL import Image
import numpy as np
sys.path.append('../..')
from data_util.lucid_dream.patchPaint import paint
from data_util.lucid_dream.lucidDream import dreamData


def gen_pairs(in_img, in_mask, in_palette, num_pairs, start_id, save_dir, stat_list, bg_img=None):
    '''
    :param in_img: input image, as cv2 image BGR order
    :param in_mask: input mask, as np.array
    :param in_palette: input palette
    :param num_pairs: how many pairs you want to generate
    :param start_id: save name of start id
    :param save_dir: the save dir
    :param stat_list: keep note of the generated data paths
    :param bg_img: whether to use an external image as the background image, default to None, otherwise cv2 BGR image
    :return: a list of generated pairs,
        each pair is of the form [img0, mask0, img1, mask1, object_id_list]
        img0, img1: jpg format
        mask0, mask1: labeled gt, pixel values are exactly the object ids
        object_id_list: list of all object ids in this pair, they are present in both img0 and img1
    '''

    if bg_img is None:
        bg = paint(in_img, np.array(in_mask), False)
        dilate_mask = True
    else:
        bg = bg_img
        dilate_mask = False

    for i in range(num_pairs):
        img1_save_name = os.path.join(save_dir, str(i+start_id).zfill(5) + 'L.jpg')
        img2_save_name = os.path.join(save_dir, str(i+start_id).zfill(5) + 'R.jpg')
        gt1_save_name = os.path.join(save_dir, str(i+start_id).zfill(5) + 'L.png')
        gt2_save_name = os.path.join(save_dir, str(i+start_id).zfill(5) + 'R.png')
        im_1, gt_1, im_2, gt_2 = dreamData(in_img, np.array(in_mask), bg, True, dilate_mask)

        # save images, rgb order on disk
        cv2.imwrite(img1_save_name, im_1)
        cv2.imwrite(img2_save_name, im_2)

        # Mask for image 1.
        gtim1 = Image.fromarray(gt_1, 'P')
        gtim1.putpalette(in_palette)
        gtim1.save(gt1_save_name)
        gtim2 = Image.fromarray(gt_2, 'P')
        gtim2.putpalette(in_palette)
        gtim2.save(gt2_save_name)

        # keep note
        stat_list.append([img1_save_name, gt1_save_name, img2_save_name, gt2_save_name])

def main():
    print('Hello')
    return 0

if __name__ == '__main__':
    main()
