'''
 Sample a number of training pairs, and save the sampled list as:
 img1_dir, gt1_dir, img2_dir, gt2_dir, obj_id

 Note that each image/mask pair may contain multiple objects in pascal12 dataset.
 There are 2913 original image/mask in pascal12 dataset.
 There are 50 pairs for each original image/mask, resulting 145650 image/mask pairs.
 However, there might be multiple objects within each image/mask pair, and we must sample the objects.

 Sampling procedure:
    for each obj_dir:                                                   --> 2913
        randomly select 10 pairs out of 50:                             --> 10
            randomly select 2 object_id if num of object >=2            --> 2 or
            select object_id=1 if num of object == 1                    --> 1

After the sampling, there will be 29130 ~ 58260 training pairs
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, glob
import numpy as np
from PIL import Image

BASE_DIR = '/storage/slurm/wangyu/PascalVOC12/'
NUM_SEG_TRAINVAL = 2913

def main():

    # TODO: get dirs, get the raw_data dir
    data_dir = os.path.join(BASE_DIR, 'Augmented')
    aug_dirs = sorted([os.path.join(data_dir, obj_dir_name) for obj_dir_name in os.listdir(data_dir) if \
                       os.path.isdir(os.path.join(data_dir, obj_dir_name))])
    if len(aug_dirs) != NUM_SEG_TRAINVAL:
        raise ValueError('Num of obj_dirs {} != num of objects {}'.format(len(aug_dirs), NUM_SEG_TRAINVAL))

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
        if len(imgsL) != 50:
            raise ValueError('Num of imgs {} != 50 for {}'.format(len(imgsL), aug_dir))

        # random permutation
        permut_order = np.random.permutation(len(imgsL))
        re_imgsL = [imgsL[i] for i in permut_order]
        re_imgsR = [imgsR[i] for i in permut_order]
        re_segsL = [segsL[i] for i in permut_order]
        re_segsR = [segsR[i] for i in permut_order]
        re_imgsL = re_imgsL[0:10]
        re_imgsR = re_imgsR[0:10]
        re_segsL = re_segsL[0:10]
        re_segsR = re_segsR[0:10]

        # for each training pair, get all object ids
        for pair_id in range(10):
            imgL, imgR, segL, segR = re_imgsL[pair_id], re_imgsR[pair_id], re_segsL[pair_id], re_segsR[pair_id]
            seg_idsL = np.unique(np.array(Image.open(segL)))
            seg_idsR = np.unique(np.array(Image.open(segR)))
            if len(seg_idsL) < 2:
                raise ValueError('Number of seg_ids {} less than 2 for {}'.format(seg_idsL, segL))
            if seg_idsL != seg_idsR:
                raise ValueError('seg_idsL {} != seg_idsR {} for {}'.format(seg_idsL, seg_idsR, segL))
            if (not 0 in seg_idsL) or (not 1 in seg_idsL):
                raise ValueError('Seg labels {} not valid for {}'.format(seg_idsL, segL))
            # get number of objects in this training pair
            num_objs = len(seg_idsL) - 1
            if num_objs == 1:
                # only a single object, check/set object_id=1, and save the pair
                if seg_idsL[-1] != 1:
                    raise ValueError('Object id {} != 1 for {}'.format(seg_idsL[-1], segL))
                sample_list.append([imgL, segL, imgR, segR, 1])
                count += 1
            else:
                # has >=2 objects within the pair, randomly pick two object_id
                obj_ids = np.random.shuffle(seg_idsL[1:])[0:2] # picking 2 object_id
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
            f.write(str(item[0]) + ' ' + str(item[1]) + ' ' + str(item[2]) + ' ' + str(item[3]) + ' ' + str(item[4]) + '\n')
        f.flush()
    print('Sampling done.')

    return 0

if __name__ == '__main__':

    main()