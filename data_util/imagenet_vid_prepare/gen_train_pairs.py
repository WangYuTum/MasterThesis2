'''
 Generate a number of files. Each file contains 1000 pairs of training images.
 Note that the script will generate all possible pairs which is a huge number.
 When generating tfrecord from those pairs, you need to use another script sample_rand_pair.py
 to generate limited number of pairs. You may specify number of pairs to sample in sample_rand_pair.py.

 File structure starting from line0:

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
SAVE_DIR = '../../tmp_data/imgnet-vid/train_pairs/'
NUM_INSTANCES = 9220
AVG_NUM_FRAMES_PER_INS = 218
PAIRS_PER_FILE = 1000 # num of pairs listed in each txt file


def write_pair_file(pair_list, pair_id):
    '''
    :param pair_list: a list where each item is a dict:
        {'img_path0':img_path0, 'img_path1':img_path1, 'bbox_0':[xmax xmin ymax ymin], 'bbox_1':[xmax xmin ymax ymin]}
    :param pair_id: file id
    :return: None
    '''

    save_path = os.path.join(SAVE_DIR, 'pair_%d'%pair_id)
    if len(pair_list) != PAIRS_PER_FILE:
        raise ValueError('Number of pairs {} in {} is not {}'.format(len(pair_list), pair_id, PAIRS_PER_FILE))
    with open(save_path, 'w') as f:
        for item in pair_list:
            f.write(item['img_path0'] + ' ' + item['img_path1'] + ' ' +
                    str(item['bbox_0'][0]) + ' ' + str(item['bbox_0'][1]) + ' ' + str(item['bbox_0'][2]) + ' ' + str(item['bbox_0'][3]) + ' ' +
                    str(item['bbox_1'][0]) + ' ' + str(item['bbox_1'][1]) + ' ' + str(item['bbox_1'][2]) + ' ' + str(item['bbox_1'][3]) + '\n')


def main():

    # get a list of instances
    ins_list = sorted([os.path.join(OBJ_INSTANCES_DIR, name) for name in os.listdir(OBJ_INSTANCES_DIR)])
    if len(ins_list) != NUM_INSTANCES:
        raise ValueError('Number of instances {} != {}'.format(len(ins_list), NUM_INSTANCES))

    # loop over all instance file
    tmp_list = []
    file_id = 0
    cnt_pair = 0
    for ins_file in ins_list:
        with open(ins_file, 'r') as f:
            print('Process {}'.format(ins_file))
            # read lines
            lines = f.read().splitlines()
            # skip 1st line
            lines = lines[1:]
            # extract all lines
            lines_items = [line.split(' ') for line in lines]
            # get all possible combinations of the current video frames
            for cur_idx in range(len(lines_items)):
                for nxt_idx in range(cur_idx+1, len(lines_items)):
                    # already accumulated 1000 pairs
                    if cnt_pair == 1000:
                        write_pair_file(tmp_list, file_id)
                        tmp_list = [] # clear tmp data
                        file_id += 1 # increase file_id
                        cnt_pair = 0 # clear counter
                    else:
                        # get cur_idx, nxt_idx to form a pair
                        left_item = lines_items[cur_idx]
                        right_item = lines_items[nxt_idx]
                        # if frame dist less than 100
                        if abs(int(left_item[0]) - int(right_item[0])) <= 100:
                            tmp_dict = {}
                            tmp_dict['img_path0'] = left_item[1]
                            tmp_dict['img_path1'] = right_item[1]
                            tmp_dict['bbox_0'] = [int(left_item[2]), int(left_item[3]), int(left_item[4]), int(left_item[5])]
                            tmp_dict['bbox_1'] = [int(right_item[2]), int(right_item[3]), int(right_item[4]), int(right_item[5])]
                            tmp_list.append(tmp_dict)
                            cnt_pair += 1
                        else:
                            continue

    print('Number of files written: {}'.format(file_id))

if __name__ == '__main__':
    main()