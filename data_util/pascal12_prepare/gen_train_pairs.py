'''
The script generates training pairs from PASCAL VOC 12 objects segmentation dataset.
For each image/mask (might contain multiple objects), apply two sets of transformations resulting in two new images/masks.
The generated two images/masks are considered as a training pair simulating two consecutive frames.

For each image/mask, we generate 10 pairs. In total, there will be 2913 * 50 = 145650 pairs. There might be multiple
objects within each pair, we must sample the objects to get a final training list. See sample_pairs.py under the same dir.
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

PASCAL_SOURCE = '/storage/slurm/wangyu/PascalVOC12/VOC2012/'
SAVE_DIR = '/storage/slurm/wangyu/PascalVOC12/Augmented/'
BASE_DIR = '/storage/slurm/wangyu/PascalVOC12/'
NUM_SEG_TRAINVAL = 2913
NUM_SEG_TRAIN = 1464

def get_dataset(data_source, trainval=True):
    '''
    :param data_source: under which: Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject
    :param trainval: get trainval or only train
    :return: a list of images/seg_gts: [(img_path0, seg_path0), (img_path1, seg_path1), ...]

    Note that we are only interested in object segmentations.
    '''

    if trainval:
        imgset = os.path.join(data_source, 'ImageSets/Segmentation/trainval.txt')
    else:
        imgset = os.path.join(data_source, 'ImageSets/Segmentation/train.txt')
    img_dir = os.path.join(data_source, 'JPEGImages')
    seg_dir = os.path.join(data_source, 'SegmentationObject')

    with open(imgset) as f:
        data_lines = f.read().splitlines()
    num_samples = len(data_lines)
    print('Got {} lines from {}'.format(num_samples, imgset))
    img_list = sorted([os.path.join(img_dir, img_name+'.jpg') for img_name in data_lines])
    seg_list = sorted([os.path.join(seg_dir, img_name+'.png') for img_name in data_lines])

    # checks
    if len(img_list) != len(seg_list):
        raise ValueError('Num of imgs {} != num of segs {}'.format(len(img_list), len(seg_list)))
    if trainval:
        if len(img_list) != NUM_SEG_TRAINVAL:
            raise ValueError('Num of imgs {} != {}'.format(len(img_list), NUM_SEG_TRAINVAL))
    else:
        if len(img_list) != NUM_SEG_TRAIN:
            raise ValueError('Num of imgs {} != {}'.format(len(img_list), NUM_SEG_TRAIN))

    # form a list of pairs
    pair_list = []
    for id in range(len(img_list)):
        pair_list.append([img_list[id], seg_list[id]])

    return pair_list

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

def main():

    # get list of img/seg pairs
    pair_list = get_dataset(PASCAL_SOURCE)
    print('Got {} samples'.format(len(pair_list)))

    palette_gen = gen_palette()
    count = 0
    stat_list = [] # keep note of all training pairs and their image/mask dirs
    for id in range(NUM_SEG_TRAINVAL):
        # read image/mask
        Iorg = cv2.imread(pair_list[id][0])
        seg_gt = Image.open(pair_list[id][1]) # object ids might already be scaled
        palette = seg_gt.getpalette()

        # check all unique object ids
        seg_gt = np.array(seg_gt)
        obj_ids = np.unique(seg_gt)
        # skip the sample if there's no 255 or 0
        if (not (0 in obj_ids)) or (not (255 in obj_ids)):
            continue
        obj_ids = np.delete(obj_ids, np.argwhere(obj_ids==255))
        # check obj id consistence
        obj_count = obj_ids[-1]
        obj_count2 = len(obj_ids) - 1
        if obj_count2 != obj_count:
            raise ValueError('obj count error in {}'.format(pair_list[id][1]))
        # replace all 255 to 0, resulting in a mask range from [0, 1, ..., max_id], (255 is empty/invalid label in pascal dataset)
        np.place(arr=seg_gt, mask=(seg_gt==255), vals=0)

        # get object color palette
        if palette is None:
            palette = palette_gen

        # generate training pairs
        print('Generating {}'.format(id))
        save_dir = os.path.join(SAVE_DIR, str(id).zfill(5))
        if os.path.isdir(save_dir) and os.path.exists(save_dir):
            gen_pairs.gen_pairs(in_img=Iorg, in_mask=seg_gt, in_palette=palette, num_pairs=50, start_id=0,
                                save_dir=save_dir, stat_list=stat_list, bg_img=None)
        else:
            os.mkdir(save_dir)
            gen_pairs.gen_pairs(in_img=Iorg, in_mask=seg_gt, in_palette=palette, num_pairs=50, start_id=0,
                                save_dir=save_dir, stat_list=stat_list, bg_img=None)
        count += 50

    # write stat_list to file
    file_name = os.path.join(BASE_DIR, 'train_pairs.txt')
    with open(file_name, 'w') as f:
        # line format: img1_dir, gt1_dir, img2_dir, gt2_dir
        f.write('line format: img1_dir anno1_dir img2_dir anno2_dir' + '\n')
        for item in stat_list:
            f.write(str(item[0])+ ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
        f.flush()


    # check length
    if count != len(stat_list):
        raise ValueError('training pair list {} != count {}'.format(len(stat_list), count))
    print('Generated {} training pairs.'.format(count))

    return 0

if __name__ == '__main__':
    main()