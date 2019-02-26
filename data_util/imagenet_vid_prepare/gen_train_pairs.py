'''
 Generate a number of files. Each file contains 1000 pairs of training images,
 file structure starting from line0:

 img_path_0 img_path_1 bbox_0 bbox_1
 ...
 ...
 img_path_2n img_path_(2n+1) bbox_2n bbox_(2n+1)


 where:
 img_path_x img_path_(x+1) : pair of images that both contain the save object
 bbox_x(xmax xmin ymax ymin): the bbox of the object in this image
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

OBJ_INSTANCES_DIR = '../../tmp_data/imgnet-vid/object_instances/'
NUM_INSTANCES = 9220


def write_pair_file(pair_list, pair_id):
    '''
    :param pair_list: a list where each item is a dict:
        {'img_path0':img_path0, 'img_path1':img_path1, 'bbox_0':[xmax xmin ymax ymin], 'bbox_1':[xmax xmin ymax ymin]}
    :param pair_id: file id
    :return: None
    '''
    pass


def main():
    pass


if __name__ == '__main__':
    main()