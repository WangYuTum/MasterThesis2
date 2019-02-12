'''
The files generates ImageNet val tfrecords.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import tensorflow as tf
import glob
from argparse import ArgumentParser
import numpy as np
from PIL import Image
import multiprocessing
import time