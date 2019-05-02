'''
    The script builds a tracker and run inference over the given video. If you want to track multiple objects in a
    single video, you have to provide multiple initial bbox. If you want to run the tracker on multiple videos, you have to
    reset the tracker for each new video. The overall procedure looks like:

    1. build a tracker
    2. init tracker with first frame image and (multiple) bbox
    3. run tracker loop for each new incoming frame
    4. reset tracker (for a new video), goto 2
    5. finish
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, glob
import tensorflow as tf
from PIL import Image
import numpy as np
from random import shuffle

os.chdir("/usr/stud/wangyu/PycharmProjects/MasterThesis2/run")
sys.path.append('..')
from core import resnet
from run import tracker
# from data_util.vot16_helper import region_to_bbox, open_img, draw_bbox
from data_util.youtube_vos_helper import get_masks, bbox_from_mask, open_img, draw_bbox, draw_bbox_mask, gen_box_colors, \
    draw_mul_bbox_mask
from data_util.davis16_helper import assembel_parts_prob, get_valid_indices, get_pre_bbox, filter_bbox_mask, \
    assembel_bbox

######### Get args
# sequence_name = 'blackswan' #  breakdance camel  dance-twirl bmx-trees dog
# frame_id = 3
init_num_bbox = int(sys.argv[3])
sequence_name = str(sys.argv[1])
frame_id = int(sys.argv[2])

############################################   Prepare dirs   ###############################################
print('Run inference on sequence {}'.format(sequence_name))
result_base = '/storage/slurm/wangyu/davis16'
result_color = os.path.join(result_base, 'results_color')
result_seg = os.path.join(result_base, 'results_seg')
result_parts = os.path.join(result_base, 'results_parts')
result_parts_color = os.path.join(result_base, 'results_parts_color')
images_base = '/usr/stud/wangyu/DAVIS17_train_val/JPEGImages/480p'
anno_base = '/usr/stud/wangyu/DAVIS17_train_val/Annotations/480p'
val_list = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
            'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
            'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

valid_path = os.path.join(result_base, 'valid_indices', sequence_name, str(frame_id - 1).zfill(5) + '.npy')
pre_bbox_path = os.path.join(result_base, 'pre_bboxes', sequence_name, str(frame_id - 1).zfill(5) + '.npy')

#########################################   Load images/gts   ##############################################
images_dir = os.path.join(images_base, sequence_name)
images_path = sorted([os.path.join(images_dir, name) for name in os.listdir(images_dir)])

annos_dir = os.path.join(anno_base, sequence_name)
annos_path = sorted([os.path.join(annos_dir, name) for name in os.listdir(annos_dir)])

init_parts_dir = os.path.join(result_parts, sequence_name, 'init_parts')
init_parts_path = sorted([os.path.join(init_parts_dir, name) for name in os.listdir(init_parts_dir)])

# get initial parts and masks
init_masks = get_masks(init_parts_path)  # list of binary masks: [[obj_id0, mask0], [obj_id1, mask1], ...] as hxwx1
init_bboxs = bbox_from_mask(init_masks)  # list of tight bboxs: [bbox0, bbox1, ...] as [xmin, ymin, xmax, ymax]
print('Got {} init parts.'.format(len(init_bboxs)))
box_colors = gen_box_colors()

# get valid indices
if frame_id == 1:
    valid_indices = list(range(len(init_bboxs)))
else:
    valid_indices = get_valid_indices(valid_path)  # list of valid box indices [0,1,4,5,6, ...]

# get the previous frame's bbox locations
if frame_id == 1:
    pre_bboxs = init_bboxs
else:
    pre_bboxs = get_pre_bbox(
        pre_bbox_path)  # list of previous bbox: [bbox0, bbox1, ...], invalid bbox has values of zeros
if len(init_bboxs) != len(pre_bboxs):
    raise ValueError('num init_bbox {} != num pre_bboxs {}'.format(len(init_bboxs), len(pre_bboxs)))

# filter parts given valid indices
new_init_bboxs = []
new_init_masks = []
new_pre_bboxs = []
for i in range(len(init_bboxs)):
    if i in valid_indices:
        new_init_bboxs.append(init_bboxs[i])
        new_init_masks.append(init_masks[i])
        new_pre_bboxs.append(pre_bboxs[i])
### Note that the init_bbox, init_masks and pre_bboxs all have lenght M <= num_init_bbox
init_masks = new_init_masks
init_bboxs = new_init_bboxs
pre_bboxs = new_pre_bboxs

# draw boxes/masks of 0th frame
img0_obj, img0_arr = open_img(images_path[0])  # img0_obj: (854, 480), img0_arr: [480, 854, 3]
if not os.path.exists(os.path.join(result_parts_color, sequence_name, 'all')):
    os.mkdir(os.path.join(result_parts_color, sequence_name, 'all'))
if frame_id == 1:
    draw_mul_bbox_mask(img_arr=img0_arr, mask_arrs=init_masks,
                       boxes=init_bboxs, colors=box_colors, indices=0,
                       save_dir=os.path.join(result_parts_color, sequence_name, 'all', '00000.jpg'))

###########################################   init tracker  ###############################################
print('Set {} templars for tracker.'.format(len(init_bboxs)))
tk = tracker.Tracker(num_templars=len(init_bboxs),  # len(init_bboxs)
                     chkp='/storage/slurm/wangyu/guide_mask/chkp/no_decoder1/youtube_vos_sgd.ckpt-312480')
input_mask_list = []
for i in range(len(init_bboxs)):
    mask_i = init_masks[i][1]  # [h, w, 1]
    input_mask_list.append(mask_i)  # [mask0, mask1, ...]
in_init_masks = np.stack(input_mask_list, 0)  # [n, h, w, 1] for n templars
tk.init_tracker(init_img=np.expand_dims(img0_arr, 0),  # [1, h, w, 3], pixel values 0-255
                init_masks=in_init_masks,
                init_bbox=init_bboxs,
                pre_bbox=pre_bboxs)  # [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
# print out tracker info
print('tracker _num_templars: {}'.format(tk._num_templars))
print('tracker _pre_locations: {}'.format(tk._pre_locations[:5]))
print('tracker _scale_templars_val: {}'.format(tk._scale_templars_val[:5]))

#########################################    tracking for frame_id    ######################################
img_obj, img_arr = open_img(images_path[frame_id])  # img_arr: [480, 854, 3]
tracked_bbox, tracked_mask = tk.track(init_img=np.expand_dims(img0_arr, 0),  # [1, 480, 854, 3]
                                      init_box=init_bboxs,
                                      search_img=np.expand_dims(img_arr, 0),  # [1, 480, 854, 3]
                                      frame_id=frame_id)  # [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
print('Track done!')
# build dirs for saving intermediate results
frame_parts_dir = os.path.join(result_parts, sequence_name, str(frame_id).zfill(5))
if not os.path.exists(frame_parts_dir):
    os.mkdir(frame_parts_dir)
else:
    file_list = [os.path.join(frame_parts_dir, name) for name in os.listdir(frame_parts_dir)]
    for file_path in file_list:
        os.remove(file_path)
frame_parts_color_dir = os.path.join(result_parts_color, sequence_name, str(frame_id).zfill(5))
if not os.path.exists(frame_parts_color_dir):
    os.mkdir(frame_parts_color_dir)
else:
    file_list = [os.path.join(frame_parts_color_dir, name) for name in os.listdir(frame_parts_color_dir)]
    for file_path in file_list:
        os.remove(file_path)

valid_idx = 0
prob_list = []
save_indices = []
save_bbox = []
for i in range(init_num_bbox):
    if i in valid_indices:
        # get the tracked bbox and mask
        prob_mask = np.squeeze(tracked_mask[valid_idx], -1)  # [480, 854], np.float32
        bin_mask = prob_mask
        bin_mask[bin_mask >= 0.65] = 1
        bin_mask[bin_mask < 0.65] = 0
        bin_mask = bin_mask.astype(np.uint8)  # [480, 854], binary mask, np.uint8
        tracked_box = tracked_bbox[valid_idx]
        prob_mask, bin_mask = filter_bbox_mask(prob_mask, bin_mask, tracked_box)
        prob_list.append(prob_mask)
        #### save tracked parts
        bin_mask_obj = Image.fromarray(bin_mask)
        bin_mask_obj.putpalette([0, 0, 0, 192, 32, 32])
        bin_mask_obj.save(os.path.join(frame_parts_dir, str(i).zfill(4) + '.png'))
        draw_bbox_mask(img_arr, np.expand_dims(bin_mask, -1), tracked_bbox[valid_idx],
                       os.path.join(frame_parts_color_dir, str(i).zfill(4) + '.jpg'), gt_bool=False)
        #### save valid indices and bbox
        save_indices.append(i)
        save_bbox.append(tracked_box)
        #### increase idx
        valid_idx += 1
    else:
        #### save dummy bbox
        save_bbox.append([0, 0, 0, 0])
        #### save zero mask if not valid
        zero_arr = np.zeros((480, 854), np.uint8)
        zero_obj = Image.fromarray(zero_arr)
        zero_obj.putpalette([0, 0, 0, 192, 32, 32])
        zero_obj.save(os.path.join(frame_parts_dir, str(i).zfill(4) + '.png'))

#### draw all bbox
if not os.path.exists(os.path.join(result_parts_color, sequence_name, 'all')):
    os.mkdir(os.path.join(result_parts_color, sequence_name, 'all'))
draw_mul_bbox_mask(img_arr=img_arr, mask_arrs=tracked_mask,
                   boxes=tracked_bbox, colors=box_colors, indices=0,
                   save_dir=os.path.join(result_parts_color, sequence_name, 'all', str(frame_id).zfill(5) + '.jpg'))

#### assemble prob_maps, and save it
print('assembling parts ...')
# final_prob, bin_mask = assembel_parts_prob(parts_list=prob_list, seq_name=sequence_name,
#                                           frame_id=frame_id, th=0.65, save_file=True)
results_parts_assemble = os.path.join(result_base, 'results_parts_assemble')
results_parts_assemble_color = os.path.join(result_base, 'results_parts_assemble_color')
assembel_bbox(img_arr=img_arr, boxes=tracked_bbox,
              sequence_name=sequence_name, frame_id=frame_id)

#### save valid indices
valid_arr = np.array(save_indices)
np.save(os.path.join(result_base, 'valid_indices', sequence_name, str(frame_id).zfill(5) + '.npy'), valid_arr)

#### save tracked bbox, which will be used in the next frame
save_bbox_arr = np.array(save_bbox)
np.save(os.path.join(result_base, 'pre_bboxes', sequence_name, str(frame_id).zfill(5) + '.npy'), save_bbox_arr)

print('Saved all results.')

"""
# tracked_bbox has len(init_bboxs) boxes
prob_list = []
for i in range(len(tracked_bbox)):
    #### prepare tracked part
    prob_mask = np.squeeze(tracked_mask[i], -1) # [480, 854], np.float32
    prob_list.append(prob_mask)
    bin_mask = prob_mask
    bin_mask[bin_mask>=0.65] = 1
    bin_mask[bin_mask<0.65] = 0
    bin_mask = bin_mask.astype(np.uint8) # [480, 854], binary mask, np.uint8
    #### save tracked parts
    bin_mask_obj = Image.fromarray(bin_mask)
    bin_mask_obj.putpalette([0,0,0,192,32,32])
    bin_mask_obj.save(os.path.join(frame_parts_dir, str(i).zfill(4) + '.png'))
    draw_bbox_mask(img_arr, np.expand_dims(bin_mask,-1), tracked_bbox[i],
                   os.path.join(frame_parts_color_dir, str(i).zfill(4) + '.jpg'), gt_bool=False)
"""
#### segmentation refinement;
