'''
Sample a number of training pairs, and save the sampled list as:
 img1_dir, gt1_dir, img2_dir, gt2_dir, obj_id

 Note that each image/mask pair may contain multiple objects in coco dataset.
 There are ??? image/mask in augmented coco.
 There are 20 pairs for each augmented image/mask, resulting ??? image/mask pairs.
 However, there might be multiple objects within each image/mask pair, and we must sample the objects.

 Sampling procedure:
    for each obj_dir:                                            --> 10725
        select all 5 pairs out of 5:                             --> 5
            select all 5 object_id                               --> 1~5

After the sampling, there will be 53625 ~ 268125 training pairs
NOTE that we must filter small bbox/mask when generating tfrecords
The object_ids within a image/mask pair might not be continuous, they also need to be dealt with when generating tfrecords

NOTE: Actually sampled pairs: 141187
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob
import numpy as np
from PIL import Image

BASE_DIR = '/storage/slurm/wangyu/coco/'
NUM_SEG_TRAIN = 10725  # num of train_pair dirs


def main():
    data_dir = os.path.join(BASE_DIR, 'Augmented')
    aug_dirs = sorted([os.path.join(data_dir, obj_dir_name) for obj_dir_name in os.listdir(data_dir) if \
                       os.path.isdir(os.path.join(data_dir, obj_dir_name))])
    if len(aug_dirs) != NUM_SEG_TRAIN:
        raise ValueError('Num of obj_dirs {} != num of expected {}'.format(len(aug_dirs), NUM_SEG_TRAIN))

    # sampling
    sample_list = []
    count = 0
    for aug_dir in aug_dirs:
        # get left/right images/segs
        imgsL = sorted(glob.glob(os.path.join(aug_dir, '*L.jpg')))
        imgsR = sorted(glob.glob(os.path.join(aug_dir, '*R.jpg')))
        segsL = sorted(glob.glob(os.path.join(aug_dir, '*L.png')))
        segsR = sorted(glob.glob(os.path.join(aug_dir, '*R.png')))

        # check length
        if len(imgsL) != len(segsL):
            raise ValueError('Num of imgsL {} != num of segsL {} for {}'.format(len(imgsL), len(segsL), aug_dir))
        if len(imgsR) != len(segsR):
            raise ValueError('Num of imgsR {} != num of segsR {} for {}'.format(len(imgsR), len(segsR), aug_dir))
        if len(imgsL) != len(imgsR):
            raise ValueError('Num of imgsL {} != num of imgsR {} for {}'.format(len(imgsL), len(imgsR), aug_dir))
        if len(imgsL) != 5:
            raise ValueError('Num of imgs {} != 5 for {}'.format(len(imgsL), aug_dir))

        # random permutation and select 5 pairs
        permut_order = np.random.permutation(len(imgsL))
        re_imgsL = [imgsL[i] for i in permut_order]
        re_imgsR = [imgsR[i] for i in permut_order]
        re_segsL = [segsL[i] for i in permut_order]
        re_segsR = [segsR[i] for i in permut_order]
        re_imgsL = re_imgsL[0:5]
        re_imgsR = re_imgsR[0:5]
        re_segsL = re_segsL[0:5]
        re_segsR = re_segsR[0:5]

        # for each training pair, get all object ids
        for pair_id in range(5):
            imgL, imgR, segL, segR = re_imgsL[pair_id], re_imgsR[pair_id], re_segsL[pair_id], re_segsR[pair_id]
            seg_idsL = np.unique(np.array(Image.open(segL)))
            seg_idsR = np.unique(np.array(Image.open(segR)))
            if not np.array_equal(seg_idsL, seg_idsR):
                # raise ValueError('seg_idsL {} != seg_idsR {} for {}'.format(seg_idsL, seg_idsR, segL))
                # get intersection over the two
                seg_idsL = list(set(seg_idsL) & set(seg_idsR))
            if not 0 in seg_idsL:
                raise ValueError('Seg labels {} not valid for {}'.format(seg_idsL, segL))
            if len(seg_idsL) < 2:
                print('Seg labels {} has object_ids < 2 for {}'.format(seg_idsL, segL))
                continue
            # pick all object_id
            obj_ids = seg_idsL[1:]
            for obj_id in obj_ids:
                sample_list.append([imgL, segL, imgR, segR, obj_id])
                count += 1

    # save sampled list
    num_samples = len(sample_list)
    print('Got {} samples.'.format(num_samples))
    file_name = os.path.join(BASE_DIR, 'sampled_pairs.txt')
    with open(file_name, 'w') as f:
        # line format: img1_dir, gt1_dir, img2_dir, gt2_dir, object_id
        f.write('line format: img1_dir anno1_dir img2_dir anno2_dir object_id' + '\n')
        for item in sample_list:
            f.write(
                str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + ' ' + str(item[4]) + '\n')
        f.flush()
    print('Sampling done.')

    return 0


if __name__ == '__main__':
    main()
