'''
The script generates training pairs using PASCAL VOC 12 as background images, ECSSD/MSRA10k as foreground objects.
Both ECSSD and MSRA10k are binary segmentation datasets, where each image only contains a single salient object.

Generation policy:
for each img/mask in ecssd/msra:
    randomly get 10 bg_imgs from pascal;
    for each bg img:
        generate 5 training pairs

Therefore, for each object, there will be 50 pairs generated
In total, 11000 * 10 * 5 = 550k training pairs
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, glob
from PIL import Image
import numpy as np
import cv2
sys.path.append('../..')
from data_util.lucid_dream import gen_pairs

PASCAL_SOURCE = '/storage/slurm/wangyu/PascalVOC12/VOC2012/'
ECSSD_SOURCE = '/storage/slurm/wangyu/ECSSD/'
MSRA10K_SOURCE = '/storage/slurm/wangyu/MSRA10k/'
SAVE_BASE_DIR = '/storage/slurm/wangyu/pascal_ecssd_mara_aug/'
NUM_BG_IMG = 17125
NUM_ECSSD = 1000
NUM_MSRA = 10000

def gen_palette():

    palette = [0,0,0,150,50,40]

    return palette

def get_pascal(data_source):
    '''
    :param data_source: under which: Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject
    :return: a list of images: [img_path0, img_path1, ...]

    Note that we only use the pascal images as backgrounds.
    '''

    img_dir = os.path.join(data_source, 'JPEGImages', '*.jpg')
    img_paths = sorted(glob.glob(img_dir))
    num_imgs = len(img_paths)
    print('Got {} background images'.format(num_imgs))

    return img_paths

def get_ecssd(data_source):
    '''
    :param data_source: under which: images, ground_truth_mask
    :return: a list of imgs/masks: [(img_path0, seg_path0), (img_path1, seg_path1), ...]

    Note that we are only interested to the object/mask
    '''

    img_dir = os.path.join(data_source, 'images', '*.jpg')
    seg_dir = os.path.join(data_source, 'ground_truth_mask', '*.png')
    img_paths = sorted(glob.glob(img_dir))
    seg_paths = sorted(glob.glob(seg_dir))
    num_imgs = len(img_paths)
    num_segs = len(seg_paths)
    if num_imgs != num_segs:
        raise ValueError('Num of imgs {} != num of segs {} for ecssd dataset.'.format(num_imgs, num_segs))
    print('Got {} images(objects) from ecssd dataset.'.format(num_imgs))

    pair_list = []
    for idx in range(num_imgs):
        pair_list.append([img_paths[idx], seg_paths[idx]])

    return pair_list

def get_msra10k(data_source):
    '''
    :param data_source:
    :return:
    '''

    img_dir = os.path.join(data_source, 'MSRA10K_Imgs_GT', 'MSRA10K_Imgs_GT', 'Imgs', '*.jpg')
    seg_dir = os.path.join(data_source, 'MSRA10K_Imgs_GT', 'MSRA10K_Imgs_GT', 'Imgs', '*.png')
    img_paths = sorted(glob.glob(img_dir))
    seg_paths = sorted(glob.glob(seg_dir))
    num_imgs = len(img_paths)
    num_segs = len(seg_paths)
    if num_imgs != num_segs:
        raise ValueError('Num of imgs {} != num of segs {} for msra10k dataset.'.format(num_imgs, num_segs))
    print('Got {} images(objects) from msra10k dataset.'.format(num_imgs))

    pair_list = []
    for idx in range(num_imgs):
        pair_list.append([img_paths[idx], seg_paths[idx]])

    return pair_list

def resize_bg(seg_gt, bg_path):
    '''
    :param seg_gt: np.array
    :param bg_path: bg image path
    :return: image of the same size as seg_gt, of the format as cv.imread()

    The function resizes the shorter side of bg_img to the corresponding size of seg_gt
    Therefore, it preserves the aspect ratio of bg_img
    '''

    mask_obj = Image.fromarray(seg_gt)
    mask_size = mask_obj.size # (width, height)
    bg_obj = Image.open(bg_path)
    bg_size = bg_obj.size # (width, height)

    shorter_side = np.minimum(bg_size[0], bg_size[1])
    if shorter_side == bg_size[0]: # width is shorter
        # get new size
        new_width = mask_size[0]
        new_height = int(bg_size[1] * (bg_size[0] / mask_size[0]))
        new_size = (new_width, new_height)
        # resize
        bg_obj = bg_obj.resize(new_size)
        if bg_obj.size[1] > mask_size[1]:
            # crop height to mask height (left, upper, right, lower)
            bg_obj = bg_obj.crop((0, 0, bg_obj.size[0], mask_size[1]))
            # convert to cv2 image in bgr order
            opencvImage = cv2.cvtColor(np.array(bg_obj), cv2.COLOR_RGB2BGR)
        else:
            # pad mean RGB to mask height
            mean_rgb = np.mean(bg_obj, axis=-1)
            opencvImage = cv2.cvtColor(np.array(bg_obj), cv2.COLOR_RGB2BGR)
            opencvImage = cv2.copyMakeBorder(opencvImage, 0, mask_size[1]-bg_obj.size[1], 0, 0, cv2.BORDER_CONSTANT, mean_rgb)

        return opencvImage
    else: # height is shorter
        # get new size
        new_height = mask_size[1]
        new_width = int(bg_size[0] * (bg_size[1] / mask_size[1]))
        new_size = (new_width, new_height)
        # resize
        bg_obj = bg_obj.resize(new_size)
        if bg_obj.size[0] > mask_size[0]:
            # crop width to mask width (left, upper, right, lower)
            bg_obj = bg_obj.crop((0, 0, mask_size[0], bg_obj.size[1]))
            # convert to cv2 image in bgr order
            opencvImage = cv2.cvtColor(np.array(bg_obj), cv2.COLOR_RGB2BGR)
        else:
            # pad mean RGB to mask width
            mean_rgb = np.mean(bg_obj, axis=-1)
            opencvImage = cv2.cvtColor(np.array(bg_obj), cv2.COLOR_RGB2BGR)
            opencvImage = cv2.copyMakeBorder(opencvImage, 0, 0, 0, mask_size[0]-bg_obj.size[0], cv2.BORDER_CONSTANT, mean_rgb)

        return opencvImage


def main():
    bg_imgs = get_pascal(PASCAL_SOURCE)
    fg_pairs0 = get_ecssd(ECSSD_SOURCE)
    fg_pairs1 = get_msra10k(MSRA10K_SOURCE)
    fg_pairs = fg_pairs0 + fg_pairs1
    if len(fg_pairs) != NUM_MSRA + NUM_ECSSD:
        raise ValueError('Total number of foreground pairs {} != {}'.format(len(fg_pairs), NUM_MSRA + NUM_ECSSD))

    palette_gen = gen_palette()
    count = 0 # count number of pairs
    stat_list = []  # keep note of all training pairs and their image/mask dirs
    for id in range(NUM_ECSSD + NUM_MSRA):
        print('Generating {}'.format(id))
        # read image/mask
        Iorg = cv2.imread(fg_pairs[id][0])
        seg_gt = Image.open(fg_pairs[id][1])  # it should be a binary mask
        palette = seg_gt.getpalette()

        # check all unique object ids
        seg_gt = np.array(seg_gt)
        obj_ids = np.unique(seg_gt)
        if len(obj_ids) != 2:  # if not having two ids
            continue
        # skip the sample if there's no 255 or 0
        if (not (0 in obj_ids)) or (not (255 in obj_ids)):
            continue
        # replace all 255 to 1, resulting in a binary 0/1 mask
        np.place(arr=seg_gt, mask=(seg_gt == 255), vals=1)

        # get object color palette
        if palette is None:
            palette = palette_gen

        # randomly shuffle bg images
        np.random.shuffle(bg_imgs)

        # for the first 10 bg images
        for bg_idx in range(10):
            # resize background image to be the same as mask/fg_img
            bg_img = resize_bg(seg_gt, bg_imgs[bg_idx])

            # generate training pairs
            save_dir = os.path.join(SAVE_BASE_DIR, 'image_pairs', str(id).zfill(5))
            if os.path.isdir(save_dir) and os.path.exists(save_dir):
                gen_pairs.gen_pairs(in_img=Iorg, in_mask=seg_gt, in_palette=palette, num_pairs=5, start_id = bg_idx * 5,
                                    save_dir=save_dir, stat_list=stat_list, bg_img=bg_img)
            else:
                os.mkdir(save_dir)
                gen_pairs.gen_pairs(in_img=Iorg, in_mask=seg_gt, in_palette=palette, num_pairs=5, start_id = bg_idx * 5,
                                    save_dir=save_dir, stat_list=stat_list, bg_img=bg_img)
            count += 5

    # write stat_list to file
    file_name = os.path.join(SAVE_BASE_DIR, 'train_pairs.txt')
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

    return 0


if __name__ == '__main__':

    main()