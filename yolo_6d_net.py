# -*- coding: utf-8 -*-
# ---------------------
# Yolo6d network, include losses
# @Author: Fan, Mo
# ---------------------

# import sys

import numpy as np
import tensorflow as tf

import config as cfg
from utils.utils import (
    confidence9,
    get_max_index,
    corner_confidences9,
    corner_confidence9,
)

class YOLO6D_net:

    def __init__(self, is_training=True):
        """
        Input images:  [batch, 416 * 416 * 3]
        Input labels:  [batch * 13 * 13 * (19 + num_classes)]
        output tensor: [batch, 13 * 13 * (19 + num_classes)]
        """
        self.is_training    = is_training
        self.Batch_Size     = cfg.BATCH_SIZE
        self.EPSILON        = cfg.EPSILON
        self.learning_rate  = cfg.LEARNING_RATE
        self.total_loss     = None
        self.disp           = cfg.DISP
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.image_size     = cfg.IMAGE_SIZE

        self.num_class  = cfg.NUM_CLASSES
        self.cell_size  = cfg.CELL_SIZE

        self.obj_scale   = cfg.CONF_OBJ_SCALE
        self.noobj_scale = cfg.CONF_NOOBJ_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.thresh      = 0.6

        self.boundry_1 = 9 * 2   ## Seperate coordinates
        self.boundry_2 = self.num_class

        self.weights = {}
        self.biases = {}
        self._init()

        self.input_images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.batch        = tf.placeholder(tf.uint8, [], name='batch')
        self.output       = self.build_networks(self.input_images)  # shape: [batch, cell, cell, 19]
        self.labels       = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 19], name='labels')

        if self.is_training:
            self.total_loss = self.loss_layer(self.output, self.labels)

# ======================== Net definition ==================================

    def build_networks(self, inputs):
        if self.disp:
            print("\n--------------Building network---------------")
        net = self.conv_layer(inputs, 'w1', 'b1', True, True, name='0_conv')

        net = self.pooling_layer(net, name='1_pool')
        net = self.conv_layer(net, 'w2', 'b2', True, True, name='2_conv')

        net = self.pooling_layer(net, name = '3_pool')
        net = self.conv_layer(net, 'w3_1', 'b3_1', True, True, name='4_conv')
        net = self.conv_layer(net, 'w3_2', 'b3_2', True, True, name='5_conv')
        net = self.conv_layer(net, 'w3_3', 'b3_3', True, True, name='6_conv')

        net = self.pooling_layer(net, name = '7_pool')
        net = self.conv_layer(net, 'w4_1', 'b4_1', True, True, name='8_conv')
        net = self.conv_layer(net, 'w4_2', 'b4_2', True, True, name='9_conv')
        net = self.conv_layer(net, 'w4_3', 'b4_3', True, True, name='10_conv')

        net = self.pooling_layer(net, name = '11_pool')
        net = self.conv_layer(net, 'w5_1', 'b5_1', True, True, name='12_conv')
        net = self.conv_layer(net, 'w5_2', 'b5_2', True, True, name='13_conv')
        net = self.conv_layer(net, 'w5_3', 'b5_3', True, True, name='14_conv')
        net = self.conv_layer(net, 'w5_4', 'b5_4', True, True, name='15_conv')
        net16 = self.conv_layer(net, 'w5_5', 'b5_5', True, True, name='16_conv')

        net = self.pooling_layer(net16, name = '17_pool')
        net = self.conv_layer(net, 'w6_1', 'b6_1', True, True, name='18_conv')
        net = self.conv_layer(net, 'w6_2', 'b6_2', True, True, name='19_conv')
        net = self.conv_layer(net, 'w6_3', 'b6_3', True, True, name='20_conv')
        net = self.conv_layer(net, 'w6_4', 'b6_4', True, True, name='21_conv')
        net = self.conv_layer(net, 'w6_5', 'b6_5', True, True, name='22_conv')
        net = self.conv_layer(net, 'w6_6', 'b6_6', True, True, name='23_conv')
        net24 = self.conv_layer(net, 'w6_7', 'b6_7', True, True, name='24_conv')

        net = self.conv_layer(net, 'w5_6', 'b5_6', True, True, name='25_conv')
        net = self.reorg(net)

        net = tf.concat([net, net24], 3)

        net = self.conv_layer(net, 'w7', 'b7', True, True, name='26_conv')
        net = self.conv_layer(net, 'w8', 'b8', False, False, name='27_conv')

        if self.disp:
            print("----------Building network complete----------\n")
        return net

    def _init(self):
        self.weights['w1'] = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01), 'w1')
        self.biases['b1']  = tf.Variable(tf.zeros([32]), 'b1')
        # pool
        self.weights['w2'] = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01), 'w2')
        self.biases['b2']  = tf.Variable(tf.zeros([64]), 'b2')
        # pool
        self.weights['w3_1'] = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), 'w3_1')
        self.biases['b3_1']  = tf.Variable(tf.zeros([128]), 'b3_1')
        self.weights['w3_2'] = tf.Variable(tf.random_normal([1, 1, 128, 64], stddev=0.01), 'w3_2')
        self.biases['b3_2']  = tf.Variable(tf.zeros([64]), 'b3_2')
        self.weights['w3_3'] = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01), 'w3_3')
        self.biases['b3_3']  = tf.Variable(tf.zeros([128]), 'b3_3')
        # pool
        self.weights['w4_1'] = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01), 'w4_1')
        self.biases['b4_1']  = tf.Variable(tf.zeros([256]), 'b4_1')
        self.weights['w4_2'] = tf.Variable(tf.random_normal([1, 1, 256, 128], stddev=0.01), 'w4_2')
        self.biases['b4_2']  = tf.Variable(tf.zeros([128]), 'b4_2')
        self.weights['w4_3'] = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=0.01), 'w4_3')
        self.biases['b4_3']  = tf.Variable(tf.zeros([256]), 'b4_3')
        # pool
        self.weights['w5_1'] = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01), 'w5_1')
        self.biases['b5_1']  = tf.Variable(tf.zeros([512]), 'b5_1')
        self.weights['w5_2'] = tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01), 'w5_2')
        self.biases['b5_2']  = tf.Variable(tf.zeros([256]), 'b5_2')
        self.weights['w5_3'] = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01), 'w5_3')
        self.biases['b5_3']  = tf.Variable(tf.zeros([512]), 'b5_3')
        self.weights['w5_4'] = tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=0.01), 'w5_4')
        self.biases['b5_4']  = tf.Variable(tf.zeros([256]), 'b5_4')
        self.weights['w5_5'] = tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=0.01), 'w5_5')
        self.biases['b5_5']  = tf.Variable(tf.zeros([512]), 'b5_5')
        self.weights['w5_6'] = tf.Variable(tf.random_normal([1, 1, 512, 64], stddev=0.01), 'w5_6')
        self.biases['b5_6']  = tf.Variable(tf.zeros([64]), 'b5_6')
        # pool
        self.weights['w6_1'] = tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01), 'w6_1')
        self.biases['b6_1'] = tf.Variable(tf.zeros([1024]), 'b6_1')
        self.weights['w6_2'] = tf.Variable(tf.random_normal([1, 1, 1024, 512], stddev=0.01), 'w6_2')
        self.biases['b6_2']  = tf.Variable(tf.zeros([512]), 'b6_2')
        self.weights['w6_3'] = tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01), 'w6_3')
        self.biases['b6_3' ]  = tf.Variable(tf.zeros([1024]), 'b6_3')
        self.weights['w6_4'] = tf.Variable(tf.random_normal([1, 1, 1024, 512], stddev=0.01), 'w6_4')
        self.biases['b6_4']  = tf.Variable(tf.zeros([512]), 'b6_4')
        self.weights['w6_5'] = tf.Variable(tf.random_normal([3, 3, 512, 1024], stddev=0.01), 'w6_5')
        self.biases['b6_5']  = tf.Variable(tf.zeros([1024]), 'b6_5')
        self.weights['w6_6'] = tf.Variable(tf.random_normal([3, 3, 1024, 1024], stddev=0.01), 'w6_6')
        self.biases['b6_6']  = tf.Variable(tf.zeros([1024]), 'b6_6')
        self.weights['w6_7'] = tf.Variable(tf.random_normal([3, 3, 1024, 1024], stddev=0.01), 'w6_7')
        self.biases['b6_7']  = tf.Variable(tf.zeros([1024]), 'b6_7')

        self.weights['w7'] = tf.Variable(tf.random_normal([3, 3, 1280, 1024], stddev=0.01), 'w7')
        self.biases['b7']  = tf.Variable(tf.zeros([1024]), 'b7')
        self.weights['w8'] = tf.Variable(tf.random_normal([1, 1, 1024, 19], stddev=0.01), 'w8')
        self.biases['b8']  = tf.Variable(tf.zeros([19]), 'b8')

    def conv_layer(self, input_, weight, bias, batch_norm, activation, name):
        """
        Args: 
            input_: input tensor, tf tensor
            weight: weight name, str
            bias: bias name, str
            batch_norm: add bn or not, bool
            activation: add leaky relu or not, bool
            name: op name
        """
        weight = self.weights[weight]
        bias = self.biases[bias]
        conv = tf.nn.conv2d(input_, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)

        if batch_norm:
            depth = weight.get_shape()[3]
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')
            conv = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
            conv = tf.add(conv, bias)
        else:
            conv = tf.add(conv, bias)

        if activation:
            return tf.nn.leaky_relu(conv, alpha=0.1)
        else:
            return conv

    def pooling_layer(self, inputs, name):
        pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
        return pool

    def reorg(self, inputs):
        """
        Reorg the tensor(half the size, 4* the depth)
        """
        outputs_1 = inputs[:, ::2, ::2, :]
        outputs_2 = inputs[:, ::2, 1::2, :]
        outputs_3 = inputs[:, 1::2, ::2, :]
        outputs_4 = inputs[:, 1::2, 1::2, :]
        output = tf.concat([outputs_1, outputs_2, outputs_3, outputs_4], axis = 3)
        return output

# ======================= Net definition end ===============================

    def loss_layer(self, inputs, labels):
        """
        Args:
            input: input tensor with shape [nB, nH, nW, 19]
            labels: label tensor with shape [nB, nH, nW, 19]
        """
        label_mask   = labels[:, :, :, 0]  # actually, it's not a conf, but a mask...[nB, nH, nW]
        label_coord = labels[:, :, :, 1:]  # [nB, nH, nW, 18]
        # one object, no class

        out_conf   = inputs[:, :, :, 0]  # [nB, nH, nW]
        out_coord  = inputs[:, :, :, 1:]  # [nB, nH, nW, 18]
        out_coord[:2] = tf.nn.sigmoid(out_coord[:2])

        label_conf, conf_mask, label_coords, out_coords = \
            self.build_target(label_coord, out_coord, label_mask)

        conf_loss  = tf.square(tf.abs(out_conf - label_conf)) * conf_mask
        conf_loss  = tf.reduce_mean(conf_loss)
        coord_loss = tf.square(tf.abs(label_coords - out_coords))
        coord_loss = tf.reduce_mean(coord_loss)
        cls_loss   = None

        return conf_loss + coord_loss + cls_loss

    def build_target(self, label, out, mask):
        """
        Args:
            label: label coords, [nB, nH, nW, 18]
            out: output coords. [nB, nH, nW, 18]
            mask: mask tells where is the object, [nB, nH, nW]
        """
        nB, nH, nW = mask.get_shape()[0], mask.get_shape()[1], mask.get_shape()[2]
        # get conf mask
        conf_mask    = []
        label_coords = []
        out_coords   = []
        for i in range(nB):
            tmp_conf_mask = tf.ones_like(mask[i]) * self.noobj_scale  # [nH, nW]
            tmp_conf_mask = mask[i] * self.obj_scale + tmp_conf_mask
            conf_mask.append(tmp_conf_mask)
            resp_x, resp_y = get_max_index(mask[i])
            label_coords.append(label[i][resp_x][resp_y])  # [18, ]
            out_coords.append(out[i][resp_x][resp_y])  # [18, ]

        conf_mask = tf.convert_to_tensor(conf_mask)  # [nB, nH, nW]
        label_coords = tf.convert_to_tensor(label_coords)  # [nB, 18]
        out_coords = tf.convert_to_tensor(out_coords)  # [nB, 18]
        # get label_conf

        return label_conf, conf_mask, label_coords, out_coords