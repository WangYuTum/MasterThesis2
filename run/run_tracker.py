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

import sys
import tensorflow as tf
sys.path.append('..')
from core import resnet

