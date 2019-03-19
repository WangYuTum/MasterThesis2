'''
Generate a single txt file, list all training pairs and their img/anno dirs,
each line has the following structure:
'img1_dir img2_dir anno1_dir anno2_dir local_object_id'

where the 'local_object_id' helps to identify the mask id of the object within this training pair

Note that each pair has maximal interval of 10, the min interval of the dataset is 5
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

OBJECTS_FILE = '/storage/slurm/wangyu/youtube_vos/train/train_objects_full.txt'
SAVE_FILE = '/storage/slurm/wangyu/youtube_vos/train/train_pairs.txt'

def main():

    with open(OBJECTS_FILE) as f:
        first_line = f.readline()
        data_lines = sorted(f.read().splitlines())
    print('Read file line format: {}'.format(first_line))
    print(len(data_lines))

    pair_data = []
    num_lines = len(data_lines)
    cur_line_idx = 0
    while cur_line_idx <= num_lines-3:
        # each frame has two pairs
        cur_line_data = data_lines[cur_line_idx].split(' ')
        pair_idx = cur_line_idx
        for _ in range(2):
            pair_idx = pair_idx + 1
            pair_line_data = data_lines[pair_idx].split(' ')
            if pair_line_data[0] != cur_line_data[0] or \
                    pair_line_data[1] != cur_line_data[1] or \
                    pair_line_data[2] != cur_line_data[2]:
                break
            else:
                if abs(int(pair_line_data[3]) - int(cur_line_data[3])) <= 10 and \
                    abs(int(pair_line_data[3]) - int(cur_line_data[3])) > 0:
                    local_obj_id = cur_line_data[2]
                    img1_dir = cur_line_data[4]
                    img2_dir = pair_line_data[4]
                    anno1_dir = cur_line_data[5]
                    anno2_dir = pair_line_data[5]
                    pair_data.append([img1_dir, img2_dir, anno1_dir, anno2_dir, local_obj_id])
                else:
                    continue
        # move pointer forward
        cur_line_idx += 1
    print('Number of pairs: {}'.format(len(pair_data)))
    with open(SAVE_FILE, 'w') as f:
        f.write('line format: img1_dir img2_dir anno1_dir anno2_dir local_object_id' + '\n')
        for line_data in pair_data:
            f.write(str(line_data[0]) + ' ' + str(line_data[1]) + ' ' + str(line_data[2]) + ' ' + str(line_data[3]) +
                    ' ' + str(line_data[4]) + '\n')
        f.flush()
    print('Write to file done: {}'.format(SAVE_FILE))

    return 0

if __name__ == '__main__':
    main()