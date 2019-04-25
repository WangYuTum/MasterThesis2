'''
    Generate object parts given:
        - Binary object mask
        - RGB image

    Properties of object parts:
        * 50 ~ 300 parts for each give object depending on the mask object size
        * object parts are square shaped, bounded by tight bbox
        * object parts bbox size range from [64, 64] ~ [256, 256]


Copyright:
    * NMS code provided from: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
from PIL import Image
import numpy as np

sys.path.append('..')
from PIL.ImageDraw import Draw
from PIL.ImageOps import expand
from random import shuffle

result_base = '/storage/slurm/wangyu/davis16'
result_parts = os.path.join(result_base, 'results_parts')
result_parts_color = os.path.join(result_base, 'results_parts_color')

images_base = '/usr/stud/wangyu/DAVIS17_train_val/JPEGImages/480p'
anno_base = '/usr/stud/wangyu/DAVIS17_train_val/Annotations/480p'

val_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
            'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
            'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

NMS_NUM_TH = 10  # only do nms if at least 10 bbox within the block
NMS_TH = 0.95  # nms threshold
IoU_TH = 0.35  # IoU threshold between part mask and object mask: 0.35
MIN_SIZE = 32
MAX_SIZE = 256


def gen_parts_seq(seq_img_dir, seq_anno_dir, seq_parts_dir, seq_color_dir):
    # prepare and check dirs
    init_parts_dir = os.path.join(seq_parts_dir, 'init_parts')
    init_color_dir = os.path.join(seq_color_dir, 'init_parts')
    img_file = os.path.join(seq_img_dir, '00000.jpg')
    anno_file = os.path.join(seq_anno_dir, '00000.png')
    if not os.path.exists(init_parts_dir):
        os.mkdir(init_parts_dir)
    else:
        # delete all init_parts before generation
        file_list = [os.path.join(init_parts_dir, name) for name in os.listdir(init_parts_dir)]
        for file_path in file_list:
            os.remove(file_path)
    if not os.path.exists(init_color_dir):
        os.mkdir(init_color_dir)
    else:
        # delete all init_parts before generation
        file_list = [os.path.join(init_color_dir, name) for name in os.listdir(init_color_dir)]
        for file_path in file_list:
            os.remove(file_path)
    if not os.path.isfile(img_file):
        raise ValueError('Image file not exists: {}'.format(img_file))
    if not os.path.isfile(anno_file):
        raise ValueError('Anno file not exists: {}'.format(anno_file))

    # generate: each binary mask part saved as xxxx.png
    num_parts = gen_parts(img_file, anno_file, init_parts_dir, init_color_dir)
    print('Generated {} parts.'.format(num_parts))

    return 0


def gen_parts(img_path, anno_path, save_path, save_color_path):
    img_obj = Image.open(img_path)  # (854, 480)
    img_arr = np.array(img_obj)  # (480, 854, 3)
    seg_obj = Image.open(anno_path)  # (854, 480)
    seg_arr = np.array(seg_obj)  # (480, 854)
    # convert seg_arr to binary
    seg_arr[seg_arr > 0] = 1

    obj_bbox = bbox_from_mask(seg_arr)  # [xmin, ymin, xmax, ymax]
    obj_h = obj_bbox[3] - obj_bbox[1]
    obj_w = obj_bbox[2] - obj_bbox[0]
    print('Object bbox size (h, w): {}'.format([obj_h, obj_w]))

    # determine block size given object bbox
    min_size = np.minimum(obj_h, obj_w)
    max_size = np.maximum(obj_h, obj_w)
    if min_size >= 300:
        block_size = 64
        MIN_SIZE = 32
        MAX_SIZE = 152
    elif min_size >= 200:
        block_size = 32
        MIN_SIZE = 28
        MAX_SIZE = 128
    elif min_size >= 100:
        block_size = 24
        MIN_SIZE = 24
        MAX_SIZE = 112
    else:
        MIN_SIZE = 24
        MAX_SIZE = 96
        block_size = 16
    print('block_size: {}'.format(block_size))
    steps_h = int(obj_h / block_size)  # num blocks in height
    steps_w = int(obj_w / block_size)  # num blocks in width
    print('Num blocks (h, w): {}'.format([steps_h, steps_w]))

    # sampling within each block
    valid_bbox = []  # where ele is [xmin, ymin, xmax, ymax]
    for block_h in range(steps_h):
        for block_w in range(steps_w):
            # get block boundary
            min_h_idx = obj_bbox[1] + block_h * block_size
            max_h_idx = 479 if min_h_idx + block_size > 479 else min_h_idx + block_size
            min_w_idx = obj_bbox[0] + block_w * block_size
            max_w_idx = 853 if min_w_idx + block_size > 853 else min_w_idx + block_size
            # get the pixel coordinates in this block that lie on the object mask
            indices = np.nonzero(seg_arr)  # (h_indices, w_indices)
            num_coord = np.shape(indices[0])[0]
            h_list = indices[0].tolist()
            w_list = indices[1].tolist()
            coor_list = []
            for i in range(num_coord):
                h_coord = h_list[i]
                w_coord = w_list[i]
                # only get points inside the current block
                if h_coord >= min_h_idx and h_coord < max_h_idx and w_coord >= min_w_idx and w_coord < max_w_idx:
                    coor_list.append([h_list[i], w_list[i]])
            # randomly sample block_size * block_size times
            sample_nums = block_size * block_size / 2
            tmp_list = []
            if len(coor_list) > 2:
                while sample_nums > 0:
                    sample_nums -= 1
                    # randomly sample a bbox size
                    bbox_size = np.random.randint(low=MIN_SIZE, high=MAX_SIZE + 1)
                    shuffle(coor_list)
                    center_h = coor_list[0][0]
                    center_w = coor_list[0][1]
                    # check if bbox is valid
                    if is_valid_bbox(seg_arr, center_h, center_w, bbox_size):
                        tmp_list.append([center_h, center_w, bbox_size])
            # do NMS within the block if num of bbox > NMS_NUM_TH
            if len(tmp_list) >= NMS_NUM_TH:
                # print('Num of bbox before NMS: {}'.format(len(tmp_list)))
                filterd_bbox = NMS_bbox(tmp_list, nms_th=NMS_TH)  # a list of bbox as [xmin, ymin, xmax, ymax]
                valid_bbox += filterd_bbox
            elif len(tmp_list) != 0:
                # convert to a list of [xmin, ymin, xmax, ymax]
                tmp_valid = []
                for item in tmp_list:
                    half_size = int(item[2] / 2)
                    x1 = item[1] - half_size
                    x2 = item[1] + half_size + 1
                    y1 = item[0] - half_size
                    y2 = item[0] + half_size + 1
                    tmp_valid.append([x1, y1, x2, y2])
                valid_bbox += tmp_valid

    num_parts = len(valid_bbox)
    print('Num of bbox after NMS: {}'.format(num_parts))
    # NMS again if number of bbox > 300
    nms_th = 0.97
    while num_parts > 400:
        tmp_list = []
        for i in range(num_parts):
            center_h = int((valid_bbox[i][3] - valid_bbox[i][1]) / 2) + valid_bbox[i][1]
            center_w = int((valid_bbox[i][2] - valid_bbox[i][0]) / 2) + valid_bbox[i][0]
            bbox_size = valid_bbox[i][3] - valid_bbox[i][1]
            tmp_list.append([center_h, center_w, bbox_size])
        valid_bbox = NMS_bbox(tmp_list, nms_th=nms_th)
        nms_th -= 0.02
        num_parts = len(valid_bbox)
        print('Num of bbox after re-NMS: {}'.format(num_parts))

    # save mask/colored
    for i in range(len(valid_bbox)):
        bbox = valid_bbox[i]  # [xmin, ymin, xmax, ymax]
        mask_arr = get_mask_from_bbox(seg_arr, bbox)  # [h,w]
        mask_obj = Image.fromarray(mask_arr)
        mask_obj.putpalette([0, 0, 0, 192, 32, 32])
        mask_obj.save(os.path.join(save_path, str(i).zfill(4) + '.png'))
        draw_bbox_mask(img_arr=img_arr, mask_arr=np.expand_dims(mask_arr, -1), bbox=bbox,
                       save_dir=os.path.join(save_color_path, str(i).zfill(4) + '.jpg'), gt_bool=True)

    return num_parts


def is_valid_bbox(seg_arr, center_h, center_w, size):
    '''
    :param seg_arr: [h, w] np.array
    :param center_h:
    :param center_w:
    :param size:
    :return: bool
    '''

    # check if bbox within image boundary
    img_size = np.shape(seg_arr)  # [h,w] = [480,854]
    img_h = img_size[0]  # 480
    img_w = img_size[1]  # 854
    half_size = int(size / 2)
    x1 = center_w - half_size
    x2 = center_w + half_size + 1
    y1 = center_h - half_size
    y2 = center_h + half_size + 1
    if x1 < 0 or y1 < 0 or x2 > img_w - 1 or y2 > img_h - 1:
        return False

    # check if IoU >= IoU_TH
    box_area = (x2 - x1) * (y2 - y1)
    part_mask = get_mask_from_bbox(seg_arr, [x1, y1, x2, y2])
    iou_val = IoU_mask(part_mask=part_mask, obj_mask=seg_arr, bbox_area=box_area)
    if iou_val < IoU_TH:
        return False
    else:
        return True


def get_mask_from_bbox(seg_arr, bbox):
    '''
    :param seg_arr: [h, w] np.array, binary mask
    :param bbox: [xmin, ymin, xmax, ymax]
    :return: [h, w] np.array, binary mask
    '''

    original_size = np.shape(seg_arr)  # [480, 854] as [height, width]
    original_h = original_size[0]
    original_w = original_size[1]
    seg_obj = Image.fromarray(seg_arr)  # (854, 480) as [width, height]
    crop_obj = seg_obj.crop(
        box=(bbox[0], bbox[1], bbox[2], bbox[3]))  # in_arg: (left, upper, right, lower), out_img: (bbox_w, bbox_h)
    pad_left = bbox[0]
    pad_top = bbox[1]
    pad_right = original_w - bbox[2]
    pad_bottom = original_h - bbox[3]
    pad_size = (pad_left, pad_top, pad_right, pad_bottom)
    paded = expand(crop_obj, border=pad_size, fill=0)

    paded_size = paded.size  # (width, height)
    if paded_size[0] != original_w or paded_size[1] != original_h:
        raise ValueError('padding part mask error, part_size: {}, seg_size: {}'.format(paded_size, seg_obj.size))

    return np.array(paded)


def NMS_bbox(bbox_list, nms_th):
    '''
    :param bbox_list: list of bbox where each ele is [center_h, center_w, bbox_size]
    :return: bbox_list after nms, each bbox is [xmin, ymin, xmax, ymax]
    '''

    tmp_list = []
    for i in range(len(bbox_list)):
        half_size = int(bbox_list[i][2] / 2)
        x1 = bbox_list[i][1] - half_size
        x2 = bbox_list[i][1] + half_size + 1
        y1 = bbox_list[i][0] - half_size
        y2 = bbox_list[i][0] + half_size + 1
        tmp_list.append([x1, y1, x2, y2])
    boxes = np.stack(tmp_list, 0)  # [N, 4] where N is the number of bbox

    # Code from Malisiewicz et al.  https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > nms_th)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    filtered = boxes[pick].astype("int")
    num_filtered = np.shape(filtered)[0]
    final_bbox_list = []
    for i in range(num_filtered):
        final_bbox_list.append(filtered[i].tolist())

    return final_bbox_list


def blend_rgb_mask(img_arr, mask):
    '''
    :param img_arr: [480, 854, 3]
    :param mask: [480, 854, 1]
    :return: Image object
    '''
    # create redish mask
    mask_r = mask * 192
    mask_g = mask * 32
    mask_b = mask * 32
    mask_rgb = np.concatenate([mask_r, mask_g, mask_b], axis=-1)
    blended = img_arr.astype(np.int32) + mask_rgb.astype(np.int32)
    blended = np.clip(blended, 0, 255)
    img_obj = Image.fromarray(blended.astype(np.uint8))

    return img_obj


def bbox_from_mask(mask):
    '''
    :param masks: [h, w] np.array
    :return: bbox as [xmin, ymin, xmax, ymax]
    '''

    indices = np.nonzero(mask)  # (h_indices, w_indices)
    min_x = indices[1].min()
    max_x = indices[1].max() + 1  # as exclusive
    min_y = indices[0].min()
    max_y = indices[0].max() + 1  # as exclusive
    bbox = [min_x, min_y, max_x, max_y]

    return bbox


def IoU_mask(part_mask, obj_mask, bbox_area):
    shape0 = np.shape(part_mask)
    shape1 = np.shape(obj_mask)
    if not np.array_equal(shape0, shape1):
        raise ValueError('Shape of part mask {} != obj mask {}'.format(shape0, shape1))

    # do element-wise multiplication to get intersection
    inters = np.sum(np.multiply(part_mask, obj_mask))
    iou_val = float(inters) / float(bbox_area)

    return iou_val


def draw_bbox_mask(img_arr, mask_arr, bbox, save_dir, gt_bool=False):
    '''
    :param img_arr: [h, w, 3]
    :param mask_arr: [h, w, 1], uint8
    :param bbox: [xmin, ymin, xmax, ymax]
    :param save_dir:
    :param gt_bool:
    :return:
    '''

    # create redish mask
    mask_r = mask_arr * 192
    mask_g = mask_arr * 32
    mask_b = mask_arr * 32
    if mask_arr.max() != 1:
        print('mask_arr sum: {}'.format(np.sum(mask_arr)))
        print('mask_max: {}'.format(mask_arr.max()))
        raise ValueError('max != 1')
    mask_rgb = np.concatenate([mask_r, mask_g, mask_b], axis=-1)  # [h,w,3], uint8
    blended = img_arr.astype(np.int32) + mask_rgb.astype(np.int32)
    blended = np.clip(blended, 0, 255)
    img_obj = Image.fromarray(blended.astype(np.uint8))

    # draw bbox
    if gt_bool:
        color = (200, 10, 10)
    else:
        color = (10, 200, 10)

    draw_handle = Draw(img_obj)
    draw_handle.rectangle(bbox, outline=color)
    img_obj.save(save_dir)


def main():
    for val_seq in val_list:
        print('Generating for seq {}'.format(val_seq))
        parts_dir = os.path.join(result_parts, val_seq)
        color_dir = os.path.join(result_parts_color, val_seq)
        img_dir = os.path.join(images_base, val_seq)
        anno_dir = os.path.join(anno_base, val_seq)
        gen_parts_seq(img_dir, anno_dir, parts_dir, color_dir)
    print('Generation done successfully!')

    return 0


if __name__ == '__main__':
    main()
