#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : tfrecords.py
#   Author      : Mofan
#   Created date: 2019-01-29 16:28:23
#   Description :
#
#================================================================

import os
import tensorflow as tf
import numpy as np
import cv2
import random

def encode_to_tfrecords(tfrecords_filename, name):
    if os.path.exists(tfrecords_filename):
        os.remove(tfrecords_filename)

    writer = tf.python_io.TFRecordWriter('./' + tfrecords_filename)

    label_path = 'LINEMOD/' + name + '/labels/'
    train_list = 'LINEMOD/' + name + '/train.txt'

    # prepare imgname, gt_labels, bg_files
    with open(train_list, 'r') as f:
        imgname = [x.strip() for x in f.readlines()]
    gt_labels = load_labels(imgname, label_path)

    bg_txt = 'VOCdevkit/VOC2012/ImageSets/Layout/trainval.txt'
    with open(bg_txt, 'r') as f:
        bg_files = [x.split()[0] for x in f.readlines()]

    for i in range(len(imgname)):
        img = imgname[i]
        # mask images
        mask_list = 'LINEMOD/' + name + '/mask/'
        mask_file = mask_list + img[-8:-3] + 'png'
        mask = cv2.imread(mask_file)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (416, 416))

        # back ground images
        bg_idx = random.randint(0, 840)
        bg_file = bg_files[bg_idx]
        bg_file_path = 'VOCdevkit/VOC2012/JPEGImages/' + bg_file + '.jpg'
        bg = cv2.imread(bg_file_path)
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
        bg = cv2.resize(bg, (416, 416))

        # object images
        obj_path = img
        obj = cv2.imread(obj_path)
        obj = cv2.cvtColor(obj, cv2.COLOR_BGR2RGB)
        obj = cv2.resize(obj, (416, 416))

        obj[mask == 0] = 0
        bg[mask == 255] = 0

        image_raw = obj + bg
        image_raw = image_raw.tostring()

        label = gt_labels[i]
        label = label_read(label, flipped=False)
        label = np.array(label)
        label = label.tostring()

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'labels': tf.train.Feature(bytes_list = tf.train.BytesList(value=[label])),
                'images': tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_raw]))
            }))
        writer.write(example.SerializeToString())

    writer.close()
    return 0


def load_labels(imgname, label_path):
    """
    Return: 2-D list, a list of all the lists in folder
    """
    gt_labels = []
    for i in range(len(imgname)):
        f_name_idx = imgname[i][-10:-4]
        f_name = f_name_idx + '.txt'
        full_path = os.path.join(label_path, f_name)
        with open(full_path, 'r') as f:
            labels = f.readline().split()
        for j in range(len(labels)):
            labels[j] = float(labels[j])
        labels[0] = int(labels[0])
        gt_labels.append(labels)
    return gt_labels

def label_read(gt_labels, flipped):
    """
    Args:
        gt_labels: a ground true label contain coordinates and class
        flipped: Whether the images are flipped
    Return:
        A 3-D tensor with shape [13, 13, 19 + num_classes]
    """
    labels = np.zeros((13, 13, 32), np.float32)

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


if __name__ == "__main__":
    train_filename = 'data/train.tfrecord'
    name = 'ape'
    encode_to_tfrecords(train_filename, name)

