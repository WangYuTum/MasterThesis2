'''
This script generates a pairs of images/masks simulating two frames from the same video, given a single image/mask.
The given image may have arbitrary number of objects labeled as [1, 2, ... M]
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from PIL import Image
import numpy as np
from patchPaint import paint
from lucidDream import dreamData


def gen_pairs(in_img, in_mask, num_pairs, bg_img=None):
    '''
    :param in_img: input image path, can be any RGB/gray image, 'jpeg'/'jpg'
    :param in_mask: input mask, may contains multiple objects, 'png'
    :param num_pairs: how many pairs you want to generate
    :param bg_img: whether to use an external image as the background image, default to None
    :return: a list of generated pairs,
        each pair is of the form [img0, mask0, img1, mask1, object_id_list]
        img0, img1: jpg format
        mask0, mask1: labeled gt, pixel values are exactly the object ids
        object_id_list: list of all object ids in this pair, they are present in both img0 and img1
    '''

    Iorg = cv2.imread(in_img)
    Morg = Image.open(in_mask) # object ids might already be scaled
    palette = Morg.getpalette()
    if palette is None:
        palette = [0, 0, 0, 150, 0, 0, 0, 150, 0, 0, 0, 150]

    if bg_img is None:
        bg = paint(Iorg, np.array(Morg), False)
        cv2.imwrite('bg.jpg', bg) # TODO: comment out
    else:
        bg = cv2.imread(bg_img)
    im_1, gt_1, im_2, gt_2 = dreamData(Iorg, np.array(Morg), bg, True)

    # save images
    # cv2.imwrite('left.jpg', im_1)
    # cv2.imwrite('right.jpg', im_1)

    # save masks
    # Mask for image 1.
    # gtim1 = Image.fromarray(gt_1, 'P')
    # gtim1.putpalette(palette)
    # gtim1.save('left.png')
    # gtim2 = Image.fromarray(gt_2, 'P')
    # gtim2.putpalette(palette)
    # gtim2.save('right.png')

    return im_1, gt_1, im_2, gt_2

def main():
    print('Hello')
    return 0

if __name__ == '__main__':
    main()
