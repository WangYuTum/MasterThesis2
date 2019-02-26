'''
 Generate statistics of ImageNet15-VID training dataset
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

OBJ_INSTANCES_DIR = '../../tmp_data/imgnet-vid/object_instances/'
SNIP_OBJECTS = '../../tmp_data/imgnet-vid/snippet_objects/'
NUM_SNIP = 4417
NUM_INSTANCES = 9220

def main():

    # get averaged num of frames per object instance
    obj_lists = sorted([os.path.join(OBJ_INSTANCES_DIR, name) for name in os.listdir(OBJ_INSTANCES_DIR)])
    if len(obj_lists) != NUM_INSTANCES:
        raise ValueError('Number of instances {} != {}'.format(len(obj_lists), NUM_INSTANCES))
    frame_cnt = 0
    for item in obj_lists:
        with open(item, 'r') as f:
            # read first line
            line_items = f.readline().split(' ')
            frame_cnt += int(line_items[1])
    avg_frames = frame_cnt / NUM_INSTANCES
    print('Average num of frames per instance: {}'.format(avg_frames))

if __name__ == '__main__':
    main()
