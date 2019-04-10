'''
The script generates training pairs using coco17 train instance segmentation dataset.
For each image/mask (might contain multiple objects), apply two sets of transformations resulting in two new images/masks.
The generated two images/masks are considered as a training pair simulating two consecutive frames.

The complete procedure:
* Get list of all images, randomly select 5000 image ids
    * for each image:
        * Extract a list of objects from an image, labels as [0, 1, 2, 3, ..., N-1]
        * Randomly select a subset (<=5) of object ids, make a new segmentation mask; -> [img, seg_mask]
        * Generate 20 pairs

We must sample the training pairs to get a final training list. See sample_pairs.py under the same dir.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
from PIL import Image
import numpy as np
import cv2
from pycocotools.coco import COCO
sys.path.append('../..')
from data_util.lucid_dream import gen_pairs

COCO_SOURCE = '/storage/slurm/wangyu/coco/'
BASE_DIR = '/storage/slurm/wangyu/coco/'
annFile = '/storage/slurm/wangyu/coco/annotations/instances_train2017.json'
imgDir = '/storage/slurm/wangyu/coco/images/'
SAVE_DIR = '/storage/slurm/wangyu/coco/Augmented/'
NUM_SEG_TRAIN = 118287
NUM_SEG_VAL = 5000
NUM_AUG_IMG = 100000

def gen_palette():
    # generate a palette of 64 colors
    num_per_ch = 4
    step_vals = [32, 96, 160, 224]
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
                if count == 64: break
            if count == 64: break
        if count == 64: break
    if len(palette) % 3 != 0:
        raise ValueError('palette not dividable by 3')
    if len(palette) / 3 > 64:
        raise ValueError('size of palette colors > 64: {}'.format(len(palette) / 3))


    return palette


def get_coco():
    '''
    :return: a list of images/seg_gts: [(img_path0, seg_path0), (img_path1, seg_path1), ...]
    '''

    print('Preparing COCO train set.')
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    print('Number of object categories: {}'.format(len(cats)))

    # get all img/ann ids
    imgIds = coco.getImgIds()  # get all img ids
    annIds = coco.getAnnIds()  # get all ann ids
    print('Number of imgIds: {}'.format(len(imgIds)))
    print('Number of annIds: {}'.format(len(annIds)))
    if len(imgIds) != NUM_SEG_TRAIN:
        raise ValueError('Num of imgIds {} != {}'.format(len(imgIds), NUM_SEG_TRAIN))

    # randomly select 5000 image ids for augmentation
    palette = gen_palette()
    count = 0
    stat_list = []  # keep note of all training pairs and their image/mask dirs
    np.random.shuffle(imgIds)
    for id in range(NUM_AUG_IMG):
        print('Generating {}'.format(id))
        # load/read image
        imgId = imgIds[id]
        img_dict = coco.loadImgs([imgId])[0]  # load image
        Iorg = cv2.imread(os.path.join(imgDir, img_dict['file_name']))

        # get anns for this image
        annIds = coco.getAnnIds(imgIds=img_dict['id'])
        anns = coco.loadAnns(annIds)

        # randomly select <=5 instances
        np.random.shuffle(anns)
        new_anns = anns[:5]
        assert len(new_anns) <= 6
        # make a new seg_mask for those instances, labeled as [0,1,2,3,4,5] where 0 is the bg
        # NOTE that individual instances might overlap
        mask = np.zeros_like(Iorg, dtype=np.uint8)[:, :, 0]
        for i, ann in enumerate(new_anns):
            mask += coco.annToMask(ann) * (i + 1)
        # checks
        obj_ids = np.unique(mask)
        if not (0 in obj_ids):
            print('No background in {}'.format(img_dict['file_name']))
            continue
        if len(obj_ids) > 6 or len(obj_ids) < 2:
            print('Num of obj_ids is {}, {} in {}'.format(len(obj_ids), obj_ids, img_dict['file_name']))
            continue
        if max(obj_ids) > 5:
            print('Max label is {} in {}'.format(max(obj_ids), img_dict['file_name']))
            continue
        obj_count = obj_ids[-1]
        obj_count2 = len(obj_ids) - 1
        if obj_count2 != obj_count:
            print('obj count error in {}'.format(img_dict['file_name']))
            continue

        # generate training pairs
        save_dir = os.path.join(SAVE_DIR, str(id).zfill(6))
        if os.path.isdir(save_dir) and os.path.exists(save_dir):
            gen_pairs.gen_pairs(in_img=Iorg, in_mask=mask, in_palette=palette, num_pairs=5, start_id=0,
                                save_dir=save_dir, stat_list=stat_list, bg_img=None)
        else:
            os.mkdir(save_dir)
            gen_pairs.gen_pairs(in_img=Iorg, in_mask=mask, in_palette=palette, num_pairs=5, start_id=0,
                                save_dir=save_dir, stat_list=stat_list, bg_img=None)
        count += 5

    # write stat_list to file
    file_name = os.path.join(BASE_DIR, 'train_pairs.txt')
    with open(file_name, 'w') as f:
        # line format: img1_dir, gt1_dir, img2_dir, gt2_dir
        f.write('line format: img1_dir anno1_dir img2_dir anno2_dir' + '\n')
        for item in stat_list:
            f.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + '\n')
        f.flush()

    # check length
    if count != len(stat_list):
        raise ValueError('training pair list {} != count {}'.format(len(stat_list), count))
    print('Generated {} training pairs.'.format(count))

def main():
    get_coco()

    return 0

if __name__ == '__main__':
    main()