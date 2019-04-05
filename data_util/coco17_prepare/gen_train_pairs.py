'''
The script generates training pairs using coco17 trainval instance segmentation dataset.
For each image/mask (might contain multiple objects), apply two sets of transformations resulting in two new images/masks.
The generated two images/masks are considered as a training pair simulating two consecutive frames.

The complete procedure:
* Extract a single object mask from each image/mask
* Apply two sets of random transformations to the image/mask 5 times

For each object, we generate 5 pairs. In total, there will be num_obj * 5 pairs.
We must sample the training pairs to get a final training list. See sample_pairs.py under the same dir.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from PIL import Image
import numpy as np
import cv2
sys.path.append('../..')
from data_util.lucid_dream import gen_pairs

COCO_SOURCE = '/storage/slurm/wangyu/coco/'
SAVE_BASE_DIR = '/storage/slurm/wangyu/coco/'
NUM_SEG_TRAINVAL = 118287 + 5000 # train + val

def gen_palette():

    # generate a palette of 255 colors
    num_per_ch = 7
    step_vals = [32, 64, 96, 128, 160, 192, 224]
    count = 0
    palette = []
    for idx_r in range(num_per_ch):
        for idx_g in range(num_per_ch):
            for idx_b in range(num_per_ch):
                tmp_r = step_vals[idx_r]
                tmp_g = step_vals[idx_g]
                tmp_b = step_vals[idx_b]
                palette.append(tmp_r)
                palette.append(tmp_g)
                palette.append(tmp_b)
                count += 1
                if count == 255: break
            if count == 255: break
        if count == 255: break
    if len(palette) % 3 != 0:
        raise ValueError('palette not dividable by 3')
    if len(palette) / 3 > 255:
        raise ValueError('size of palette colors > 255: {}'.format(len(palette)/3))


    return palette

def get_coco(data_source):
    '''
    :param data_source: under which: train2017, val2017, annotations
    :return: a list of images/seg_gts: [(img_path0, seg_path0), (img_path1, seg_path1), ...]
    '''


    # form a list of pairs
    pair_list = []
    for id in range(len(img_list)):
        pair_list.append([img_list[id], seg_list[id]])

    return pair_list

def main():

    # get list of img/seg pairs
    pair_list = get_coco(COCO_SOURCE)
    print('Got {} samples'.format(len(pair_list)))

    return 0

if __name__ == '__main__':
    main()