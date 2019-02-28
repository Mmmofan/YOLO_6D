# -*- coding: utf-8 -*-
# ---------------------
# configuration for yolo-6d
# @Author: Fan, Mo
# @Email: fmo@nullmax.ai
# ---------------------

import os

"""
Path and Dataset parameters
"""

DISP = True

##Files parameters
DATA_DIR     = 'data'
CACHE_DIR    = os.path.join(DATA_DIR, 'cache')
CACHE_FILE   = os.path.join(CACHE_DIR, 'yolo_best.ckpt')
OUTPUT_DIR   = os.path.join(DATA_DIR, 'output')
WEIGHTS_DIR  = os.path.join(DATA_DIR, 'weights')
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, 'yolo_6d.ckpt')

FLIPPED = True

##Network parameters
NUM_CLASSES  = 13
BATCH_SIZE   = 4
WEIGHT_DECAY = 0.0005
EPSILON      = 1e-5
OPTIMIZER    = 'ADAMS'
IMAGE_SIZE   = 416
CHANNELS     = 3
BATCH_NORM   = True

ALPHA = 2.0
Dth   = 80.0

CELL_SIZE = 13
NUM_COORD = 18

BOXES_PER_CELL   = 1
CONF_OBJ_SCALE   = 4.9
CONF_NOOBJ_SCALE = 0.1
CLASS_SCALE      = 1.0
COORD_SCALE      = 1.0

#Training parameters
GPU = '2'
# LEARNING_RATE = 0.001
PRE_BOUNDARIES = [1, 50, 1000, 2000, 8000]
BOUNDARIES     = [1, 50, 1000, 4000, 10000]
LEARNING_RATE  = [0.0001, 0.0001, 0.001, 0.0001, 0.00001, 0.000001]
DECAY_STEP     = 10000
DECAY_RATE     = 0.1
STAIRCASE      = True

EPOCH = 200

SUMMARY_ITER = 10
SAVE_ITER    = 50

#Test parameters
CONF_THRESHOLD = 0.1
NMS_THRESHOLD  = 0.4
IOU_THRESHOLD  = 0.5
