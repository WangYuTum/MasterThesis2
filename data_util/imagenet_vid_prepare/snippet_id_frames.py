'''
Generate a single txt file, list all snippets and the number of frames they have,
each line has the following structure:
'snippet_id snippet_dir number_frames'

Example (given dir_root as: ILSVRC2015/Data/VID/train/):
    '0 dir_root/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000 100'
Example (given dir_root as: ILSVRC2015/Data/VID/val/):
    '0 dir_root/ILSVRC2015_val_00000000 100'
The corresponding annotations can be found by replacing 'Data' with 'Annotations'
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

TRAIN_DIR = '/storage/slurm/wangyu/imagenet15_vid/ILSVRC2015/Data/VID/train/'
VAL_DIR = '/storage/slurm/wangyu/imagenet15_vid/ILSVRC2015/Data/VID/val/'
SAVE_FILE = '../../tmp_data/imgnet-vid/snip_id_frames.txt'

def main():

    # read train/val dirs
    train_lv1 = sorted([os.path.join(TRAIN_DIR, subdir) for subdir in os.listdir(TRAIN_DIR)])
    # read each snippet in train dir
    snippets_train = []
    for lv1 in train_lv1:
        snippets_train += sorted([os.path.join(lv1, subdir) for subdir in os.listdir(lv1)])
    # read each snippet in val dir
    snippets_val = sorted([os.path.join(VAL_DIR, subdir) for subdir in os.listdir(VAL_DIR)])
    snippets = snippets_train + snippets_val

    print('Number of train snippets: {}'.format(len(snippets_train)))
    print('Number of val snippets: {}'.format(len(snippets_val)))
    print('Number of total snippets: {}'.format(len(snippets)))

    # write to file
    print('Writing file to {}'.format(SAVE_FILE))
    with open(SAVE_FILE, 'w') as f:
        for id, snip_dir in enumerate(snippets):
            num_frames = len([name for name in os.listdir(snip_dir) if os.path.isfile(os.path.join(snip_dir, name))])
            f.write(str(id) + ' ' + snip_dir + ' ' + str(num_frames) + '\n')
        f.flush()
    print('Write to file complete.')

if __name__ == '__main__':
    main()


