# -*- coding: utf-8 -*-
# ---------------------
# using Linemod dataset for training and testing
# @Author: Fan, Mo
# @Email: fmo@nullmax.ai
# ---------------------

import os
import random
import cv2
import numpy as np

import yolo.config as cfg
from utils.utils import *


class Linemod(object):

    def __init__(self, phase, arg=None):
        # Set parameters for training and testing
        self.data_options = read_data_cfg(arg)
        self.trainlist = self.data_options['train']
        self.testlist = self.data_options['valid']
        self.meshname = self.data_options['mesh']
        self.backupdir = self.data_options['backup']
        self.diam = float(self.data_options['diam'])
        self.dataset_name = self.data_options['name']
        self.vx_threshold = self.diam * 0.1

        self.phase = phase
        self.datasets_dir = os.path.join('LINEMOD', self.dataset_name)
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.image_width = 640   # axis x
        self.image_height = 480   # axis y
        self.mask_path = 'LINEMOD/' + self.dataset_name + '/mask/'
        self.mask_files = None
        self.bg_txt = 'VOCdevkit/VOC2012/ImageSets/Layout/trainval.txt'
        self.bg_files = None

        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.num_classes = cfg.NUM_CLASSES
        self.flipped = False
        self.train_imgname = None
        self.test_imgname = None
        self.imgname = None
        self.train_gt_labels = None
        self.test_gt_labels = None
        self.gt_labels = None
        self.batch = 0
        print("\n---------------Loading dataset---------------")
        self.prepare(self.phase)  # get the image files name and label files name
        # print(len(self.bg_files))
        print("----------Loading dataset complete-----------\n")

    def prepare(self, phase):
        """
        self.imgname: A list of all training image files
        self.gt_labels: A list of all ground true labels(which elements are lists like [[1,xx,xx,...],[1,xx,xx,...]] with integer and float numbers)
        these two matched respectively
        """
        if phase == 'train':
            with open(self.trainlist, 'r') as f:
                self.imgname = [x.strip() for x in f.readlines()]  # a list of trianing files
            self.gt_labels = self.load_labels() # a list of all labels with respect to imgname

            for ro, _, fi in os.walk(self.mask_path):
                root, __, files = ro, _, fi
            self.mask_files = files

            with open(self.bg_txt, 'r') as f:
                self.bg_files = [x.split()[0] for x in f.readlines()]

        elif phase == 'test':
            with open(self.testlist, 'r') as f:
                self.imgname = [x.strip() for x in f.readlines()]
            self.gt_labels = self.load_labels()

        else:
            print('\n   Wrong phase...\n   Try again...')

    def next_batches(self):
        images = np.zeros((self.batch_size, 416, 416, 3), np.float32)
        labels = np.zeros((self.batch_size, 13, 13, 1 + self.boxes_per_cell*9*2 + self.num_classes), np.float32)

        # for idx in range(self.batch_size):
            # images[idx] = self.image_read(self.imgname[idx + self.batch * self.batch_size], self.flipped)
            # labels[idx] = self.label_read(self.gt_labels[idx + self.batch * self.batch_size], self.flipped)
        for idx in range(self.batch_size):
            images[idx] = self.image_bg_replace(self.imgname[idx+self.batch*self.batch_size][-8:-3], self.flipped)
            labels[idx] = self.label_read(self.gt_labels[idx + self.batch * self.batch_size], self.flipped)

        self.batch += 1
        return images, labels

    def next_batches_test(self):
        images = np.zeros((self.batch_size, 416, 416, 3), np.float32)
        labels = np.zeros((self.batch_size, 13, 13, 32), np.float32)

        for idx in range(self.batch_size):
            images[idx] = self.image_bg_replace(self.imgname[idx+self.batch*self.batch_size][-8:-3], self.flipped)
            labels[idx] = self.label_read(self.gt_labels[idx + self.batch * self.batch_size], self.flipped)

        return images, labels

    def get_truths(self):
        gt_truths = []
        for idx in range(self.batch_size):
            gt_truths.append(self.gt_labels[idx + self.batch * self.batch_size])
        return gt_truths

    def load_labels(self):
        """
        Return: 2-D list, a list of all the lists in folder
        """
        gt_labels = []
        label_path = os.path.join(self.datasets_dir, 'labels')
        for i in range(len(self.imgname)):
            f_name_idx = self.imgname[i][-10:-4]
            f_name = f_name_idx + '.txt'
            full_path = os.path.join(label_path, f_name)
            with open(full_path, 'r') as f:
                labels = f.readline().split()
            for j in range(len(labels)):
                labels[j] = float(labels[j])
            labels[0] = int(labels[0]) * 1.0
            gt_labels.append(labels)
        return gt_labels

    def image_bg_replace(self, imgname, flipped):
        imgname += 'png'
        mask_path = self.mask_path + imgname
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (self.image_size, self.image_size))

        rand_num = random.randint(0, 840)
        bg_file = self.bg_files[rand_num]
        bg_file_path = 'VOCdevkit/VOC2012/JPEGImages/' + bg_file + '.jpg'
        bg = cv2.imread(bg_file_path)
        bg = cv2.resize(bg, (self.image_size, self.image_size))

        obj_path = 'LINEMOD/' + self.dataset_name + '/JPEGImages/' + '00' + imgname[:-3] + 'jpg'
        obj = cv2.imread(obj_path)
        obj = cv2.resize(obj, (self.image_size, self.image_size))

        obj[mask == 0] = 0
        bg[mask == 255] = 0
        res = obj + bg

        cv2.imwrite("replaced.jpg", res)
        #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB).astype(np.float32)
        res = (res / 255.0) * 2.0 - 1.0

        if flipped:
            res = cv2.flip(res, 0)

        return res

    def image_read(self, imgname, flipped):
        image = cv2.imread(imgname)
        image = cv2.resize(image, (self.image_size, self.image_size))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0

        if flipped:
            image = cv2.flip(image, 0)

        return image

    def label_read(self, gt_labels, flipped):
        """
        Args:
            gt_labels: a ground true label contain coordinates and class
            flipped: Whether the images are flipped
        Return:
            A 3-D tensor with shape [13, 13, 19 + num_classes]
        """
        labels = np.zeros((13, 13, 1+self.boxes_per_cell*9*2 + self.num_classes), np.float32)

        gt_label = gt_labels[0]
        gt_xc = gt_labels[1]  * 13
        gt_yc = gt_labels[2]  * 13
        gt_x0 = gt_labels[3]  * 13
        gt_y0 = gt_labels[4]  * 13
        gt_x1 = gt_labels[5]  * 13
        gt_y1 = gt_labels[6]  * 13
        gt_x2 = gt_labels[7]  * 13
        gt_y2 = gt_labels[8]  * 13
        gt_x3 = gt_labels[9]  * 13
        gt_y3 = gt_labels[10] * 13
        gt_x4 = gt_labels[11] * 13
        gt_y4 = gt_labels[12] * 13
        gt_x5 = gt_labels[13] * 13
        gt_y5 = gt_labels[14] * 13
        gt_x6 = gt_labels[15] * 13
        gt_y6 = gt_labels[16] * 13
        gt_x7 = gt_labels[17] * 13
        gt_y7 = gt_labels[18] * 13

        if not flipped:
            coords = [gt_xc, gt_yc, gt_x0, gt_y0, gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3,
                      gt_x4, gt_y4, gt_x5, gt_y5, gt_x6, gt_y6, gt_x7, gt_y7]
        else:
            coords = [gt_xc, 13-gt_yc, gt_x7, 13-gt_y7, gt_x6, 13-gt_y6, gt_x5, 13-gt_y5, gt_x4, 13-gt_y4,
                      gt_x3, 13-gt_y3, gt_x2, 13-gt_y2, gt_x1, 13-gt_y1, gt_x0, 13-gt_y0]

        response_x = int(gt_xc)
        response_y = int(gt_yc)

        # set response value to 1
        labels[response_x, response_y, 0] = 1.0

        # set coodinates value
        for i in range(1, 19, 1):
            if i % 2 != 0: # x
                labels[response_x, response_y, i] = coords[i - 1] - response_x
            else: # y
                labels[response_x, response_y, i] = coords[i - 1] - response_y

        # set label
        labels[response_x, response_y, 19 + gt_label] = 1

        return labels

