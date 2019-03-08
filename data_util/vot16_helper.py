from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
from PIL.ImageDraw import Draw


def region_to_bbox(region, center=True):
    n = len(region)
    assert n == 4 or n == 8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return cx, cy, w, h
    else:
        # region[0] -= 1
        # region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return cx, cy, w, h
    else:
        return cx - w / 2, cy - h / 2, w, h

def open_img(img_path):
    img_obj = Image.open(img_path)
    img_arr = np.array(img_obj)

    return img_obj, img_arr

def draw_bbox(img_obj, bbox, save_dir, gt_bool=False):
    '''
    :param img_obj: PIL Image Object
    :param bbox: [xmin, ymin, xmax, ymax]
    :return:
    '''
    if gt_bool:
        color = (200, 10, 10)
    else:
        color = (10, 200, 10)

    draw_handle = Draw(img_obj)
    draw_handle.rectangle(bbox, outline=color)

    img_obj.save(save_dir)
    print('Draw bbox {}'.format(save_dir))
