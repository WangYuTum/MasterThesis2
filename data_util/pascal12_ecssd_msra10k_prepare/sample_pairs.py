'''
 Sample a number of training pairs, and save the sampled list as:
 img1_dir, gt1_dir, img2_dir, gt2_dir, obj_id

 Note that each image/mask pair only contains single object in pascal12_ecssd_msra10k dataset. There are 11000 objects,
 for each object there are 50 augmented pairs. By default, for each object, we only sample 5 pairs, resulting to
 5 * 11000 = 55k training pairs

 Sampling procedure:
    for each obj_dir:
        pick 5 pairs randomly, check/set obj_id=1
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob
import numpy as np
from PIL import Image

BASE_DIR = '/storage/slurm/wangyu/pascal_ecssd_mara_aug/'
NUM_OBJS = 11000

def main():

    # TODO: get dirs, get raw_data dirs
    data_dir = os.path.join(BASE_DIR, 'raw_data')
    obj_dirs = sorted([os.path.join(data_dir, obj_dir_name) for obj_dir_name in os.listdir(data_dir) if \
                os.path.isdir(os.path.join(data_dir, obj_dir_name))])
    if len(obj_dirs) != NUM_OBJS:
        raise ValueError('Num of obj_dirs {} != num of objects {}'.format(len(obj_dirs), NUM_OBJS))

    # sampling
    sample_list = []
    for obj_dir in obj_dirs:
        # get left/right images/segs
        imgsL = sorted(glob.glob(os.path.join(obj_dir, '*L.jpg')))
        imgsR = sorted(glob.glob(os.path.join(obj_dir, '*R.jpg')))
        segsL = sorted(glob.glob(os.path.join(obj_dir, '*L.png')))
        segsR = sorted(glob.glob(os.path.join(obj_dir, '*R.png')))

        # check length
        if len(imgsL) != len(segsL):
            raise ValueError('Num of imgsL {} != num of segsL {} for {}'.format(len(imgsL), len(segsL), obj_dir))
        if len(imgsR) != len(segsR):
            raise ValueError('Num of imgsR {} != num of segsR {} for {}'.format(len(imgsR), len(segsR), obj_dir))
        if len(imgsL) != len(imgsR):
            raise ValueError('Num of imgsL {} != num of imgsR {} for {}'.format(len(imgsL), len(imgsR), obj_dir))
        if len(imgsL) != 50:
            raise ValueError('Num of imgs {} != 50 for {}'.format(len(imgsL), obj_dir))

        # random permutation
        permut_order = np.random.permutation(len(imgsL))
        re_imgsL = [imgsL[i] for i in permut_order]
        re_imgsR = [imgsR[i] for i in permut_order]
        re_segsL = [segsL[i] for i in permut_order]
        re_segsR = [segsR[i] for i in permut_order]
        re_imgsL = re_imgsL[0:5]
        re_imgsR = re_imgsR[0:5]
        re_segsL = re_segsL[0:5]
        re_segsR = re_segsR[0:5]

        # get samples
        for i in range(5):
            # check object id label
            seg_idsL = np.unique(np.array(Image.open(re_segsL[i])))
            seg_idsR = np.unique(np.array(Image.open(re_segsR[i])))
            if seg_idsL != seg_idsR:
                raise ValueError('seg_idsL {} != seg_idsR {} for {}'.format(seg_idsL, seg_idsR, re_segsL[i]))
            if len(seg_idsL) != 2 or (not 0 in seg_idsL) or (not 1 in seg_idsL):
                raise ValueError('Seg labels {} not valid for {}'.format(seg_idsL, re_segsL[i]))
            # img1_dir, gt1_dir, img2_dir, gt2_dir, obj_id
            sample_list.append([re_imgsL[i], re_segsL[i], re_imgsR[i], re_segsR[i], 1])

    # save sampled list
    num_samples = len(sample_list)
    if num_samples != 55000:
        raise ValueError('Num of samples {} != {}'.format(num_samples, 55000))
    file_name = os.path.join(BASE_DIR, 'sampled_pairs.txt')
    with open(file_name, 'w') as f:
        # line format: img1_dir, gt1_dir, img2_dir, gt2_dir, object_id
        f.write('line format: img1_dir anno1_dir img2_dir anno2_dir object_id' + '\n')
        for item in sample_list:
            f.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + ' ' + str(item[4]) + '\n')
        f.flush()
    print('Sampling done.')

    return 0

if __name__ == '__main__':

    main()