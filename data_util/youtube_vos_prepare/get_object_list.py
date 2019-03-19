'''
Generate a single txt file, list all objects and their corresponding video ids and frames,
each line has the following structure:
'global-object-id video-id local-object-id frame-id'
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os, json

TRAIN_DIR = '/storage/slurm/wangyu/youtube_vos/train/'
SAVE_FILE = '/storage/slurm/wangyu/youtube_vos/train/train_objects.txt'
SAVE_FILE_FULL = '/storage/slurm/wangyu/youtube_vos/train/train_objects_full.txt'

def main():

    meta_file = TRAIN_DIR + 'meta.json'
    with open(meta_file) as f:
        meta_data = json.load(f)

    meta_data = meta_data['videos']
    cur_oid = 0
    object_list = []
    for vid in meta_data:
        cur_vid = vid
        obs_dict = meta_data[vid]['objects']
        for obj_id in obs_dict:
            frame_list = obs_dict[obj_id]['frames']
            for frame_id in frame_list:
                save_list = [cur_oid, cur_vid, obj_id, frame_id]
                object_list.append(save_list)
            cur_oid += 1
    print('Number of total object instances: {}'.format(len(object_list)))
    # write to file
    """
    with open(SAVE_FILE, 'w') as f:
        f.write('global-object-id video-id local-object-id frame-id' + '\n')
        for inst in object_list:
            f.write(str(inst[0]) + ' ' + str(inst[1]) + ' ' + str(inst[2]) + ' ' + str(inst[3]) + '\n')
        f.flush()
    print('Write file done: {}'.format(SAVE_FILE))
    """
    with open(SAVE_FILE_FULL, 'w') as f:
        f.write('global-object-id video-id local-object-id frame-id img-dir anno-dir' + '\n')
        for inst in object_list:
            img_dir = TRAIN_DIR + 'JPEGImages/' + str(inst[1]) + '/' + str(inst[3]) + '.jpg'
            anno_dir = TRAIN_DIR + 'Annotations/' + str(inst[1]) + '/' + str(inst[3]) + '.png'
            f.write(str(inst[0]) + ' ' + str(inst[1]) + ' '+ str(inst[2]) + ' ' + str(inst[3]) + ' ' + img_dir + ' ' + anno_dir + '\n')
        f.flush()
    print('Write file done: {}'.format(SAVE_FILE_FULL))

    return 0

if __name__ == '__main__':
    main()

