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
from linemod import Linemod
from utils.MeshPly import MeshPly
from utils.timer import Timer
from utils.utils import *
from yolo.yolo_6d_net import YOLO6D_net

class Detector(object):

    def __init__(self, net, data, weights_file):
        self.yolo           = net
        self.data           = data
        self.num_classes    = cfg.NUM_CLASSES
        self.image_size     = cfg.IMAGE_SIZE
        self.cell_size      = cfg.CELL_SIZE
        self.batch_size     = cfg.BATCH_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold      = cfg.CONF_THRESHOLD
        self.categories     = ['ape', 'benchvise', 'cam', 'can', 'cat', 'driller', 'duck',
                               'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone']

        self.variable_to_restore = tf.global_variables()
        self.restorer            = tf.train.Saver(self.variable_to_restore)
        self.sess                = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print("-------------restoring weights file from {}---------------".format(weights_file))
        self.restorer.restore(self.sess, weights_file)

    def detect(self):
        all_images = self.data.imgname
        all_labels = self.data.gt_labels
        assert(len(all_images)==len(all_labels))
        # for i in range(len(all_images)):
        for i in range(10):
            image_path   = all_images[i]
            gt_label     = all_labels[i]
            image, label = self.data_read(image_path, gt_label)
            w, h, d      = image.shape[0], image.shape[1], image.shape[2]
            input_image  = np.reshape(image, [1, w, h, d])
            feed_dict    = {self.yolo.input_images: input_image}
            output       = self.sess.run(self.yolo.logit, feed_dict=feed_dict)  # 4-D [1, 13, 13, 32]
            self.post_process(output, image_path, i)
        return

    def post_process(self, output, image_path, number):
        coords = output[:, :, :, :18] # [batch, 13, 13, 18]
        class_prob = output[:, :, :, 18:-1]  # [batch, 13, 13, 13]
        confidence = output[:, :, :, -1]  # [batch, 13, 13]
        coords = np.concatenate([sigmoid_func(coords[:,:,:,:2]), coords[:,:,:,2:]], axis=3)
        # class_prob = softmax(class_prob)

        boxes = []
        for i in range(confidence.shape[0]):
            conf = confidence[i]  # 2-D [13, 13]
            max_conf = np.max(conf)
            idxi, idxj = np.where(conf == max_conf)
            idxi, idxj = idxi[0], idxj[0]
            classes = class_prob[i, idxi, idxj, :]
            #classes = softmax(classes)
            max_class_val= np.max(classes)
            class_id = np.where(classes==max_class_val)
            coord = coords[i, idxi, idxj, :]
            xc = coord[0]  + idxi
            yc = coord[1]  + idxj
            x1 = coord[2]  + idxi
            y1 = coord[3]  + idxj
            x2 = coord[4]  + idxi
            y2 = coord[5]  + idxj
            x3 = coord[6]  + idxi
            y3 = coord[7]  + idxj
            x4 = coord[8]  + idxi
            y4 = coord[9]  + idxj
            x5 = coord[10] + idxi
            y5 = coord[11] + idxj
            x6 = coord[12] + idxi
            y6 = coord[13] + idxj
            x7 = coord[14] + idxi
            y7 = coord[15] + idxj
            x8 = coord[16] + idxi
            y8 = coord[17] + idxj
            box = [xc*32.,yc*32.,x1*32.,y1*32.,x2*32.,y2*32.,x3*32.,y3*32.,x4*32.,y4*32.,
                    x5*32.,y5*32.,x6*32.,y6*32.,x7*32.,y7*32.,x8*32.,y8*32.,class_id]
            boxes.append(box)

        for box in boxes:
            self.draw(box, image_path, number)

        print('image showed')

    def draw(self, box, image_path, number):
        image = cv2.imread(image_path)
        xc, yc, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8 = \
                int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4]), int(box[5]), \
                int(box[6]), int(box[7]), int(box[8]), int(box[9]), int(box[10]), int(box[11]),\
                int(box[12]), int(box[13]), int(box[14]), int(box[15]), int(box[16]), int(box[17])
        class_id = box[18][0][0]
        assert(class_id>=0 and class_id<=13)
        name = 'draw_' + str(number) + self.categories[class_id] + '.jpg'

        cv2.circle(image, (xc, yc), 2, (255, 0, 0), 1)
        cv2.line(image, (x1, y1), (x3, y3), (0,0,255), 2)
        cv2.line(image, (x1, y1), (x5, y5), (0,0,255), 2)
        cv2.line(image, (x3, y3), (x7, y7), (0,0,255), 2)
        cv2.line(image, (x5, y5), (x7, y7), (0,0,255), 2)

        cv2.line(image, (x2, y2), (x4, y4), (0,0,255), 2)
        cv2.line(image, (x2, y2), (x6, y6), (0,0,255), 2)
        cv2.line(image, (x8, y8), (x4, y4), (0,0,255), 2)
        cv2.line(image, (x8, y8), (x6, y6), (0,0,255), 2)

        cv2.line(image, (x1, y1), (x2, y2), (0,0,255), 2)
        cv2.line(image, (x3, y3), (x4, y4), (0,0,255), 2)
        cv2.line(image, (x5, y5), (x6, y6), (0,0,255), 2)
        cv2.line(image, (x7, y7), (x8, y8), (0,0,255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = 'center: (' + str(xc) + ',' + str(yc) + ')'
        image = cv2.putText(image, text, (20, 20), font, 0.6, (0,0,255), 1)
        cv2.imwrite(name, image)

    def data_read(self, img, lbl):
        image = cv2.imread(img)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0

        label = self.label_read(lbl)

        return image, label

    def label_read(self, gt_labels):
        """
        Args:
            gt_labels: a ground true label contain coordinates and class
            flipped: Whether the images are flipped
        Return:
            A 3-D tensor with shape [13, 13, 19 + num_classes]
        """
        classes = np.zeros((13, 13, self.num_classes), np.float32)

        gt_label = gt_labels[0]
        gt_xc    = gt_labels[1]  * 13.0
        gt_yc    = gt_labels[2]  * 13.0
        gt_x0    = gt_labels[3]  * 13.0
        gt_y0    = gt_labels[4]  * 13.0
        gt_x1    = gt_labels[5]  * 13.0
        gt_y1    = gt_labels[6]  * 13.0
        gt_x2    = gt_labels[7]  * 13.0
        gt_y2    = gt_labels[8]  * 13.0
        gt_x3    = gt_labels[9]  * 13.0
        gt_y3    = gt_labels[10] * 13.0
        gt_x4    = gt_labels[11] * 13.0
        gt_y4    = gt_labels[12] * 13.0
        gt_x5    = gt_labels[13] * 13.0
        gt_y5    = gt_labels[14] * 13.0
        gt_x6    = gt_labels[15] * 13.0
        gt_y6    = gt_labels[16] * 13.0
        gt_x7    = gt_labels[17] * 13.0
        gt_y7    = gt_labels[18] * 13.0

        
        coords = [0.0, gt_xc, gt_yc, gt_x0, gt_y0, gt_x1, gt_y1, gt_x2, gt_y2, gt_x3, gt_y3,
                  gt_x4, gt_y4, gt_x5, gt_y5, gt_x6, gt_y6, gt_x7, gt_y7]

        response_x = int(gt_xc)
        response_y = int(gt_yc)

        coords = np.array(coords).reshape(1, 1, -1)
        coords = np.tile(coords, (13, 13, 1))  # [13, 13, 19]
        

        # set response value to 1
        coords[response_x, response_y, 0] = 1.0

        # set label
        classes[response_x, response_y, int(gt_label)] = 1  # [13, 13, classes]

        labels = np.concatenate([coords, classes], 2)

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
