#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : valid.py
#   Author      : Mofan
#   Created date: 2019-01-25 10:44:31
#   Description :
#
#================================================================


from __future__ import print_function

import argparse
import os

import numpy as np
import tensorflow as tf

import yolo.config as cfg
from pascal_voc import Pascal_voc
from linemod import Linemod
from utils.MeshPly import MeshPly
from utils.timer import Timer
from utils.utils import *
from yolo.yolo_6d_net import YOLO6D_net

class Detector(object):

    def __init__(self, net, data, weights_file):
        self.yolo = net
        self.data = data
        self.num_classes = cfg.NUM_CLASSES
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.CONF_THRESHOLD
        self.categories = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
                           'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']

        self.variable_to_restore = tf.global_variables()
        self.restorer = tf.train.Saver(self.variable_to_restore)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("-------------restoring weights file from {}---------------".format(weights_file))
        self.restorer.restore(self.sess, weights_file)

    def detect(self):
        all_images = self.data.imgname
        all_labels = self.data.gt_labels
        acc = []
        for i in range(len(all_images)):
            image_path = all_images[i]
            gt_label = all_labels[i]
            image, label = self.data_read(image_path, gt_label)
            w, h, d = image.shape[0], image.shape[1], image.shape[2]
            input_image = np.reshape(image, [1, w, h, d])
            feed_dict = {self.yolo.input_images: input_image}
            output = self.sess.run(self.yolo.logit, feed_dict=feed_dict)
            self.post_process(output, image_path)

    def post_process(self, input, image_path):
        coords = input[:, :, :, :18]
        class_prob = input[:, :, :, 18:-1]
        confidence = input[:, :, :, -1].reshape(-1,coords.shape[1],coords.shape[2], 1)
        coords = np.concatenate([sigmoid_func(coords[:,:,:,:2]), coords[:,:,:,2:]], axis=3)
        # class_prob = softmax(class_prob)

        confs = confidence[:, :, :, -1]
        boxes = []
        for i in range(confs.shape[0]):
            conf = confs[i]
            idxi, idxj = np.where(conf == np.max(conf))
            idxi, idxj = idxi[0], idxj[0]
            # idxi, idxj = 5, 5
            classes = class_prob[i,idxi,idxj,:]
            classes = softmax(classes)
            class_id = np.where(classes==np.max(classes))
            coord = coords[i, idxi, idxj, :].reshape(9,2)
            xc = coord[0,0] + idxi
            yc = coord[0,1] + idxj
            x1 = coord[1,0] + idxi
            y1 = coord[1,1] + idxj
            x2 = coord[2,0] + idxi
            y2 = coord[2,1] + idxj
            x3 = coord[3,0] + idxi
            y3 = coord[3,1] + idxj
            x4 = coord[4,0] + idxi
            y4 = coord[4,1] + idxj
            x5 = coord[5,0] + idxi
            y5 = coord[5,1] + idxj
            x6 = coord[6,0] + idxi
            y6 = coord[6,1] + idxj
            x7 = coord[7,0] + idxi
            y7 = coord[7,1] + idxj
            x8 = coord[8,0] + idxi
            y8 = coord[8,1] + idxj
            box = [xc/13,yc/13,x1/13,y1/13,x2/13,y2/13,x3/13,y3/13,x4/13,y4/13,
                    x5/13,y5/13,x6/13,y6/13,x7/13,y7/13,x8/13,y8/13,class_id]
            boxes.append(box)

        for box in boxes:
            self.draw(box, image_path)
        
        print('image showed')

    def draw(self, box, image_path):
        image = cv2.imread(image_path)
        xc, yc, x1, y1, x2, y2, x3, y3, x4, y4 = int(box[0]*416), int(box[1]*416), int(box[2]*416), int(box[3]*416), int(box[4]*416), \
                                                  int(box[5]*416), int(box[6]*416), int(box[7]*416), int(box[8]*416), int(box[9]*416)
        x5, y5, x6, y6, x7, y7, x8, y8 = int(box[10]*416), int(box[11]*416), int(box[12]*416), int(box[13]*416), int(box[14]*416), int(box[15]*416), int(box[16]*416), int(box[17]*416)

        cv2.circle(image, (xc, yc), 3, (255, 0, 0), 1)
        cv2.line(image, (x1, y1), (x3, y3), (255,0,0), 2)
        cv2.line(image, (x1, y1), (x5, y5), (255,0,0), 2)
        cv2.line(image, (x3, y3), (x7, y7), (255,0,0), 2)
        cv2.line(image, (x5, y5), (x7, y7), (255,0,0), 2)

        cv2.line(image, (x2, y2), (x4, y4), (0,255,0), 2)
        cv2.line(image, (x2, y2), (x6, y6), (0,255,0), 2)
        cv2.line(image, (x8, y8), (x4, y4), (0,255,0), 2)
        cv2.line(image, (x8, y8), (x6, y6), (0,255,0), 2)
        cv2.imwrite('draw.jpg', image)

    def data_read(self, img, lbl):
        image = self.image_read(img)
        label = self.label_read(lbl)

        return image, label

    def image_read(self, imgname):
        image = cv2.imread(imgname)
        image = cv2.resize(image, (self.image_size, self.image_size))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0

        return image

    def label_read(self, gt_labels):

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

        coords = [gt_xc, gt_yc, gt_x0, gt_y0, gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3,
                    gt_x4, gt_y4, gt_x5, gt_y5, gt_x6, gt_y6, gt_x7, gt_y7]

        response_x = int(gt_xc)
        response_y = int(gt_yc)

        # set response value to 1
        labels[response_x, response_y, 0] = 1

        # set coodinates value
        for i in range(1, 19, 1):
            if i % 2 != 0: # x
                labels[response_x, response_y, i] = coords[i - 1] - response_x
            else: # y
                labels[response_x, response_y, i] = coords[i - 1] - response_y

        # set label
        labels[response_x, response_y, 19 + gt_label] = 1

        return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datacfg', default='cfg/ape.data', type=str)
    parser.add_argument('--weights', default='yolo_6d.ckpt', type=str)
    parser.add_argument('--gpu', default= '', type=str)
    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISABLE_DEVICES'] = args.gpu
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)

    yolo = YOLO6D_net(is_training=False)
    data = Linemod('test', args.datacfg)
    detector = Detector(yolo, data, weight_file)

    detector.detect()
