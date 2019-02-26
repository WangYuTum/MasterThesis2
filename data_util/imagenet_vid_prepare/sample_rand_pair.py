'''
 Generate a number of files. Each file contains 1000 pairs of training images.
 Note that the script will take the outputs from gen_train_pairs.py and randomly sample a number of pairs
 from them. You must specify the number of pairs to be sampled.

 Recommend sampling at least 2500000 pairs (50 eps x 50000)

 Generated file structure starting from line0:
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
from random import shuffle

NUM_SAMPLES = 4000000 # 80 eps x 50000 = 4M
PAIR_DIR = '../../tmp_data/imgnet-vid/train_pairs/'
SAVE_DIR = '../../tmp_data/imgnet-vid/sampled_pairs/'
NUM_TRIAN_PAIR_FILES = 157398
NUM_INSTANCES = 9220
AVG_NUM_FRAMES_PER_INS = 218
PAIRS_PER_FILE = 1000 # num of pairs listed in each txt file
NUM_GEN_FILES = int(NUM_SAMPLES / PAIRS_PER_FILE)


def write_pair_file(pair_list, pair_id):
    '''
    :param pair_list: a list where each item is a dict:
        {'img_path0':img_path0, 'img_path1':img_path1, 'bbox_0':[xmax xmin ymax ymin], 'bbox_1':[xmax xmin ymax ymin]}
    :param pair_id: file id
    :return: None
    '''
    save_path = os.path.join(SAVE_DIR, 'pair_%d'%pair_id)
    if len(pair_list) < PAIRS_PER_FILE:
        raise ValueError('Number of pairs {} in {} is less than {}'.format(len(pair_list), pair_id, PAIRS_PER_FILE))
    shuffle(pair_list)
    with open(save_path, 'w') as f:
        for id in range(PAIRS_PER_FILE):
            item = pair_list[id]
            f.write(item['img_path0'] + ' ' + item['img_path1'] + ' ' +
                    str(item['bbox_0'][0]) + ' ' + str(item['bbox_0'][1]) + ' ' + str(item['bbox_0'][2]) + ' ' + str(item['bbox_0'][3]) + ' ' +
                    str(item['bbox_1'][0]) + ' ' + str(item['bbox_1'][1]) + ' ' + str(item['bbox_1'][2]) + ' ' + str(item['bbox_1'][3]) + '\n')
    print('Write file {}'.format(save_path))


def main():

    # get list of files
    pair_files = sorted([os.path.join(PAIR_DIR, name) for name in os.listdir(PAIR_DIR)])
    shuffle(pair_files)
    num_files = len(pair_files)
    print('Got {} files'.format(num_files))

    # get number of sampled pairs per file
    samples_per_file = int(NUM_SAMPLES / num_files) + 1
    print('Sample {} pairs from each file.'.format(samples_per_file))
    files_to_sample = int(num_files / NUM_GEN_FILES)
    print('Files to be sampled per {}'.format(files_to_sample))
    divide = files_to_sample * NUM_GEN_FILES
    remainder = int(num_files % NUM_GEN_FILES)

    # build sample list
    sample_list = []
    step = files_to_sample
    for idx in range(NUM_GEN_FILES):
        sample_list.append([idx*step, (idx+1)*step])
    # distribute remainders
    for remaind_id in range(remainder):
        if divide + remaind_id < num_files:
            sample_list[remaind_id] = sample_list[remaind_id] + [divide + remaind_id]

    # sample pairs
    for file_id in range(NUM_GEN_FILES):
        tmp_list = []
        # sample
        current_list = sample_list[file_id] # either [start, end] or [start, end, remainder]
        for idx in range(current_list[0], current_list[1]):
            file_dir = pair_files[idx]
            # sample samples_per_file pairs from this file
            with open(file_dir, 'r') as f:
                lines = f.read().splitlines()
                shuffle(lines)
                for cnt_pair in range(samples_per_file):
                    line_items = lines[cnt_pair].split(' ')
                    tmp_dict = {}
                    tmp_dict['img_path0'] = line_items[0]
                    tmp_dict['img_path1'] = line_items[1]
                    tmp_dict['bbox_0'] = [int(line_items[2]), int(line_items[3]), int(line_items[4]), int(line_items[5])]
                    tmp_dict['bbox_1'] = [int(line_items[6]), int(line_items[7]), int(line_items[8]),
                                          int(line_items[9])]
                    tmp_list.append(tmp_dict)
        if len(current_list) == 3:
            with open(pair_files[current_list[2]], 'r') as f:
                lines = f.read().splitlines()
                shuffle(lines)
                for cnt_pair in range(samples_per_file):
                    line_items = lines[cnt_pair].split(' ')
                    tmp_dict = {}
                    tmp_dict['img_path0'] = line_items[0]
                    tmp_dict['img_path1'] = line_items[1]
                    tmp_dict['bbox_0'] = [int(line_items[2]), int(line_items[3]), int(line_items[4]), int(line_items[5])]
                    tmp_dict['bbox_1'] = [int(line_items[6]), int(line_items[7]), int(line_items[8]),
                                          int(line_items[9])]
                    tmp_list.append(tmp_dict)
        # write to file
        write_pair_file(tmp_list, file_id)

if __name__ == '__main__':
    main()