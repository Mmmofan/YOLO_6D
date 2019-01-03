import copy
import os

import cv2
import numpy as np
import tensorflow as tf

import yolo.config as cfg
from utils.utils import *


class Linemod(object):

    def __init__(self, phase, arg=None):
        # Set parameters for training and testing
        self.data_options = read_data_cfg(arg)
        self.trainlist = self.data_options['train']
        self.testlist = self.data_options['test']
        self.gpus = self.data_options['gpu']
        self.meshname = self.data_options['mesh']
        self.num_workers = int(self.data_options['num_workers'])
        self.backupdir = self.data_options['backup']
        self.diam = float(self.data_options['diam'])
        self.vx_threshold = self.diam * 0.1

        self.phase = phase
        self.datasets_dir = cfg.DATASETS_DIR
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.num_classes = cfg.NUM_CLASSES
        self.flipped = False
        self.imgname = None
        self.gt_labels = None
        self.epoch = 0
        self.batch = 0
        self.prepare()  # get the image files name and label files name


    def next_batches(self):
        images = np.zeros((self.batch_size, 416, 416, 3), np.float32)
        labels = np.zeros((self.batch_size, 13, 13, 32), np.float32)
        for idx in range(self.batch_size):
            images[idx] = self.image_read(self.imgname[idx + self.epoch * self.batch_size])
            labels[idx] = self.label_read(self.imgname[idx + self.epoch * self.batch_size])
        self.epoch += 1
        return images, labels

    def load_labels(self):
        #if self.phase == 'train':
            return

    def image_read(self, imgname, flipped=False):
        image = cv2.imread(imgname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    #def label_read(self, imgname):

    def prepare(self):
        with open(self.trainlist, 'r') as f:
            self.imgname = [x.strip() for x in f.readlines()]  # a list of trianing files
        self.gt_labels = self.load_labels()
