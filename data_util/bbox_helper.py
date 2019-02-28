'''
    Some of the bbox operations used
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def draw_bbox_templar(templars, bbox_templars, batch):
    '''
    :param templars: [batch, 3, 256, 256]
    :param bbox_templars: [batch, 4], [xmin, ymin, xmax, ymax]
    :param batch: batch size
    :return: drawn templars [batch, 256, 256, 3]
    '''

    # [batch, num_bounding_boxes, 4]
    batch_bbox_list = [] # [[2,4], [2,4], ...]
    for batch_i in range(batch):
        # get tight bbox
        tmp_box = tf.cast(bbox_templars[batch_i], tf.float32)
        tight_bbox = [tmp_box[1] / 256.0, tmp_box[0] / 256.0, tmp_box[3] / 256.0, tmp_box[2] / 256.0] # [y_min, x_min, y_max, x_max]
        # get central bbox
        central_bbox = [0.25, 0.25, 0.75, 0.75]
        # stack
        batch_i_bbox = tf.stack(values=[central_bbox, tight_bbox], axis=0) # [2, 4]
        batch_bbox_list.append(batch_i_bbox)
    # stack batch
    batch_bbox = tf.stack(values=batch_bbox_list, axis=0) # [batch, 2, 4]
    drawed_images = tf.image.draw_bounding_boxes(images=tf.transpose(templars, [0,2,3,1]),
                                                 boxes=batch_bbox,
                                                 name='draw_temp_bbox')

    return drawed_images

def draw_bbox_search(searchs, bbox_searchs, batch):
    '''
    :param searchs: [batch, 3, 256, 256]
    :param bbox_searchs: [batch, 4], [xmin, ymin, xmax, ymax]
    :param batch:  batch size
    :return: drawn searchs [batch, 256, 256, 3]
    '''

    batch_bbox_list = []  # [[1,4], [1,4], ...]
    for batch_i in range(batch):
        # get tight bbox
        tmp_box = tf.cast(bbox_searchs[batch_i], tf.float32)
        tight_bbox = [tmp_box[1] / 256.0, tmp_box[0] / 256.0, tmp_box[3] / 256.0,
                      tmp_box[2] / 256.0]  # [y_min, x_min, y_max, x_max]
        tight_bbox = tf.expand_dims(tight_bbox, axis=0) # [1, 4]
        batch_bbox_list.append(tight_bbox)
    # stack batch
    batch_bbox = tf.stack(values=batch_bbox_list, axis=0)  # [batch, 1, 4]
    drawed_images = tf.image.draw_bounding_boxes(images=tf.transpose(searchs, [0, 2, 3, 1]),
                                                 boxes=batch_bbox,
                                                 name='draw_search_bbox')

    return drawed_images