'''
    The file implements cross-correlation bbox tracking & mask_prop train data pipeline.

    Some functions/methods are directly copied from tensorflow official resnet model:
    https://github.com/tensorflow/models/tree/r1.12.0/official/resnet

    Therefore the code must be used under Apache License, Version 2.0 (the "License"):
    http://www.apache.org/licenses/LICENSE-2.0
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
from scipy.ndimage.morphology import binary_dilation

_DEFAULT_SIZE = 256
_NUM_CHANNELS = 3
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
_NUM_TRAIN = 285849 # number of training pairs, equal to number of sampled pairs
_NUM_SHARDS = 256 # number of tfrecords in total, each tfrecord has 1116 pairs, only one has 1269 pairs

