'''
 Generate a file for each object instance (train/val), the file lists all images that contain the
 object and their corresponding bboxes, frame ids. The file structure starting from line0:

 object_id num_frames width height
 frame_id0 img_path0 xmax xmin ymax ymin
 frame_id1 img_path1 xmax xmin ymax ymin
 ...
 ...
 ...
 frame_idn img_pathn xmax xmin ymax ymin


 object_id: unique global object id in ImageNet15-VID dataset
 frame_idx: the frame id within the current snippet
 img_pathx: the image that contains the object
 bbox(xmax xmin ymax ymin): the bbox of the object in this image
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

SNIP_OBJECTS = '../../tmp_data/imgnet-vid/snippet_objects/'
SAVE_DIR = '../../tmp_data/imgnet-vid/object_instances/'
NUM_SNIP = 4417


def write_object_file(object_dict, obj_id):
    '''
    :param object_dict:
        'width': width
        'height': height
        'num_frames': num_frames
        'data_list': [data0, data1, ..., datan]
            datax = {'frame_id':frame_id, 'img_path':img_path, 'bbox':[xmax xmin ymax ymin]}
    :return:
    '''
    file_dir = os.path.join(SAVE_DIR, 'obj_%d'%obj_id)

    with open(file_dir, 'w') as f:
        # write head info
        f.write(str(obj_id) + ' ' + str(object_dict['num_frames']) + ' ' + str(object_dict['width']) + ' ' + str(object_dict['height']) + '\n')
        # write list of data
        data_list = object_dict['data_list']
        for item in data_list:
            f.write(str(item['frame_id']) + ' ' + item['img_path'] + ' ' +
                    str(item['bbox'][0]) + ' ' + str(item['bbox'][1]) + ' ' +
                    str(item['bbox'][2]) + ' ' + str(item['bbox'][3]) + '\n')
    print('Wrote {}'.format(file_dir))

def main():

    # get all snippet files
    snippet_list = sorted([os.path.join(SNIP_OBJECTS, name) for name in os.listdir(SNIP_OBJECTS)])
    if len(snippet_list) != NUM_SNIP:
        raise ValueError('Number of snippets {} != {}'.format(len(snippet_list), NUM_SNIP))

    # for each snippet file, read object info
    object_cnt = 0
    for snip_dir in snippet_list:
        print('Process {}'.format(snip_dir))
        with open(snip_dir, 'r') as snip_f:
            obj_dict = {}
            lines = snip_f.read().splitlines()
            # get object and associated info, split chars: '#####'
            for line in lines:
                line_items = line.split(' ')
                if len(line_items) == 1: # either start of a new object or end of the current object
                    if line_items[0] == '#####': # end of the current object
                        write_object_file(obj_dict, object_cnt)
                        object_cnt += 1
                    else: # encounter a new object
                        # clear the tmp dict
                        obj_dict = {}
                        obj_dict['width'] = 0
                        obj_dict['height'] = 0
                        obj_dict['num_frames'] = 0
                        obj_dict['data_list'] = []
                else:
                    obj_dict['width'] = int(line_items[6])
                    obj_dict['height'] = int(line_items[7])
                    obj_dict['num_frames'] += 1
                    tmp_dict = {}
                    tmp_dict['frame_id'] = int(line_items[0])
                    tmp_dict['img_path'] = line_items[1]
                    tmp_dict['bbox'] = [int(line_items[2]), int(line_items[3]), int(line_items[4]), int(line_items[5])]
                    obj_dict['data_list'].append(tmp_dict)

if __name__ == '__main__':
    main()
