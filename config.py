# -*- coding: utf-8 -*-
# ---------------------
# configuration for yolo-6d
# @Author: Fan, Mo
# Email: fmo@nullmax.ai
# ---------------------

import os

"""
Path and Dataset parameters
"""

DISP = True

##Files parameters
DATA_DIR = 'data'
DATASETS_DIR = os.path.join(DATA_DIR, 'datasets')
CACHE_DIR = os.path.join(DATASETS_DIR, 'cache')
OUTPUT_DIR = os.path.join(DATASETS_DIR, 'output')
WEIGHTS_DIR = os.path.join(DATASETS_DIR, 'weights')
WEIGHTS_FILE = os.path.join(DATA_DIR, 'weights', 'yolo_6d.ckpt')

##Network parameters
BATCH_SIZE = 32
TEST_BATCH_SIZE =64
WEIGHT_DECAY = 0.0005
MAX_PAD = 'SAME'
EPSILON = 1e-5
OPTIMIZER = 'ADAMS'
IMAGE_SIZE = 416
CHANNELS = 3
BATCH_NORM = True

ALPHA = 2.0
Dth = 30.0   ###30 pixels

CELL_SIZE = 13
NUM_COORD = 18

BOXES_PER_CELL = 1
CONF_OBJ_SCALE = 4.9
CONF_NOOBJ_SCALE = 0.1
CLASS_SCALE = 1.0
COORD_SCALE = 1.0

#Training parameters
GPU = '0'
LEARNING_RATE = 0.001
DECAY_STEP = 3200  ## 64 * 100
DECAY_RATE = 0.1
STAIRCASE = True
MAX_ITER = 15000
SUMMARY_ITER = 10
SAVE_ITER = 1000

#Test parameters
THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
