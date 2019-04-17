'''
    The script builds a tracker and run inference over the given video. If you want to track multiple objects in a
    single video, you have to specify multiple bbox. If you want to run the tracker on multiple videos, you have to
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
sys.path.append('..')
from core import resnet
from run import tracker
from data_util.vot16_helper import region_to_bbox, open_img, draw_bbox
from data_util.youtube_vos_helper import get_masks, bbox_from_mask

sequence_name = '0c7a4680db'


######### load images/gt_box ##########

# get ready files
video_dir0 = '/storage/slurm/wangyu/youtube_vos/valid/JPEGImages/' + sequence_name
save_dir = '/work/wangyu/youtube_vos/valid/results/no_decoder1/' + sequence_name
gt_file = '/storage/slurm/wangyu/youtube_vos/valid/Annotations/' + sequence_name + '/00040.png'  # segmentation mask
frame_list = sorted(glob.glob(os.path.join(video_dir0, '*.jpg')))  # images

# get ready gts
gt_list = []
gt_masks = get_masks(gt_file)  # list of binary masks: [mask0, mask1, ...] as hxwx1
gt_bboxs = bbox_from_mask(gt_masks)  # list of tight bboxs: [bbox0, bbox1, ...] as [xmin, ymin, xmax, ymax]
for i in range(len(gt_masks)):
    gt_list.append([gt_masks[i], gt_bboxs[i]])  # a list of gt pairs: [(mask0, bbox0), (mask1, bbox1), ...]
print('num of test frames: {}, number of gts: {}'.format(len(frame_list), len(gt_list)))

# TODO
gt0 = region_to_bbox(gt_list[0], center=False) # [xmin, ymin, w, h]
# draw gt0 bbox and save
print('Init bbox positions: {}'.format([gt0[0], gt0[1], gt0[0]+gt0[2], gt0[1]+gt0[3]]))
img0_obj, img0_arr = open_img(frame_list[0])
draw_bbox(img0_obj, [gt0[0], gt0[1], gt0[0]+gt0[2], gt0[1]+gt0[3]], save_dir+'/00000001.jpg', gt_bool=True) # [xmin, ymin, xmax, ymax]


######### init tracker ###########
trajectory = []
trajectory.append([gt0[0], gt0[1], gt0[0]+gt0[2], gt0[1]+gt0[3]])
tk = tracker.Tracker(num_templars=1, chkp='/storage/slurm/wangyu/imagenet15_vid/chkp/imgnetvid_4gpu_sgd2/imgnetvid_4gpu.ckpt-143750')
sum_writer = tf.summary.FileWriter(logdir=save_dir, graph=tk._sess.graph)
init_box = [[gt0[0], gt0[1], gt0[0]+gt0[2], gt0[1]+gt0[3]]]
tk.init_tracker(init_img=np.expand_dims(img0_arr, 0), # [1, h, w, 3], pixel values 0-255
                init_bbox=init_box) #[[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
# print out tracker info
print('tracker _num_templars: {}'.format(tk._num_templars))
print('tracker _pre_locations: {}'.format(tk._pre_locations))
print('tracker _scale_templars_val: {}'.format(tk._scale_templars_val))


######## tracking loop: one frame at a time ###########
num_frames = len(frame_list)
#for frame_id in range(1, num_frames):
for frame_id in range(1, 60):
    img_obj, img_arr = open_img(frame_list[frame_id])
    tracked_bbox, response, sum_op_ = tk.track(init_img=np.expand_dims(img0_arr, 0),
                                               init_box=init_box,
                                               search_img=np.expand_dims(img_arr, 0),
                                               frame_id=frame_id) # [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax], ...]
    trajectory.append(tracked_bbox[0])
    draw_bbox(img_obj, tracked_bbox[0], save_dir + '/' + str(frame_id+1) + '.jpg', gt_bool=False)
    #response *= 255.0
    #Image.fromarray(response.astype(np.uint8)).save(save_dir + '/' + str(frame_id+1) + '_response.jpg')
    sum_writer.add_summary(sum_op_, frame_id)
    sum_writer.flush()
sum_writer.close()
