# -*- coding: utf-8 -*-
# ---------------------
# Yolo6d network, include losses
# @Author: Fan, Mo
# @Email: fmo@nullmax.ai
# ---------------------

import sys

import numpy as np
import tensorflow as tf

import yolo.config as cfg
from utils.utils import *


"""
5BTM 51C9 3791 5TKL
This is the script to try to reproduce YOLO-6D
input: a mini-batch of images
train:

"""

class YOLO6D_net:

    Batch_Size = cfg.BATCH_SIZE
    WEIGHT_DECAY = cfg.WEIGHT_DECAY
    MAX_PADDING = cfg.MAX_PAD
    EPSILON = cfg.EPSILON
    learning_rate = cfg.LEARNING_RATE
    optimizer = None
    total_loss = None
    disp = cfg.DISP
    param_num = 0
    boxes_per_cell = cfg.BOXES_PER_CELL
    image_size = cfg.IMAGE_SIZE

    num_class = cfg.NUM_CLASSES
    Batch_Norm = cfg.BATCH_NORM
    ALPHA = cfg.ALPHA
    cell_size = cfg.CELL_SIZE
    num_coord = cfg.NUM_COORD  ## 18: 9 points, 8 corners + 1 centroid

    obj_scale = cfg.CONF_OBJ_SCALE
    noobj_scale = cfg.CONF_NOOBJ_SCALE
    class_scale = cfg.CLASS_SCALE
    coord_scale = cfg.COORD_SCALE

    def __init__(self, is_training=True):
        """
        Input images: [416 * 416 * 3]
        output tensor: [13 * 13 * (18 + 1 + num_classes)]
            self.input_images ==> self.logit
        Input labels: [batch * 13 * 13 * 20 + num_classes]
        """
        self.boundry_1 = 9 * 2 * self.boxes_per_cell   ## Seperate coordinates
        self.boundry_2 = self.num_class
        
        #off_set:  [self.cell_size, self.cell_size, 18]
        self.off_set = np.transpose(np.reshape(np.array(
                                    [np.arange(self.cell_size)] * self.cell_size * 18 * self.boxes_per_cell),
                                    (18, self.cell_size, self.cell_size)),
                                    (1, 2, 0))

        self.input_images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='Input')
        self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 18 + 1 + self.num_class], name='Labels')

        self.logit = self._build_net(self.input_images)
        self.confidence = None

        if is_training:
            self.loss_layer(self.logit, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('Total loss', self.total_loss)

        #self.conf_value = tf.reshape(self.logit[:, :, :, -1], [-1, self.cell_size, self.cell_size, 1])
        self.conf_score = self.confidence_score(self.logit, self.confidence)


    def _build_net(self, input):
        if self.disp:
            print("--------Building network---------")
        self.Batch_Norm = True
        x = self.conv(input, 3, 1, 32, 'leaky', name='0_conv')
        x = self.max_pool_layer(x, name='1_pool')

        x = self.conv(x, 3, 1, 64, 'leaky', name='2_conv')
        x = self.max_pool_layer(x, name='3_pool')

        x = self.conv(x, 3, 1, 128, 'leaky', name='4_conv')
        x = self.conv(x, 1, 1, 64, 'leaky', name='5_conv')
        x = self.conv(x, 3, 1, 128, 'leaky', name='6_conv')
        x = self.max_pool_layer(x, name='7_pool')

        x = self.conv(x, 3, 1, 256, 'leaky', name='8_conv')
        x = self.conv(x, 1, 1, 128, 'leaky', name='9_conv')
        x = self.conv(x, 3, 1, 256, 'leaky', name='10_conv')
        x = self.max_pool_layer(x, name='11_pool')
        
        x = self.conv(x, 3, 1, 512, 'leaky', name='12_conv')
        x = self.conv(x, 1, 1, 256, 'leaky', name='13_conv')
        x = self.conv(x, 3, 1, 512, 'leaky', name='14_conv')
        x = self.conv(x, 1, 1, 256, 'leaky', name='15_conv')
        x_16 = self.conv(x, 3, 1, 512, 'leaky', name='16_conv')
        x = self.max_pool_layer(x_16, name='17_pool')

        x = self.conv(x, 3, 1, 1024, 'leaky', name='18_conv')
        x = self.conv(x, 1, 1, 512, 'leaky', name='19_conv')
        x = self.conv(x, 3, 1, 1024, 'leaky', name='20_conv')
        x = self.conv(x, 1, 1, 512, 'leaky', name='21_conv')
        x = self.conv(x, 3, 1, 1024, 'leaky', name='22_conv')

        x = self.conv(x, 3, 1, 1024, 'leaky', name='23_conv')
        x = self.conv(x, 3, 1, 1024, 'leaky', name='24_conv')

        x_ps = self.conv(x_16, 1, 1, 64, 'leaky', name='25_conv')
        x_ps = self.reorg(x_ps)
        
        x = tf.concat([x, x_ps], 3)

        x = self.conv(x, 3, 1, 1024, 'leaky', name='26_conv')
        self.Batch_Norm = False
        x = self.conv(x, 1, 1, 18 + 1 + self.num_class, 'linear', name='27_conv') ## 9 points 1 confidence C classes

        if self.disp:
            print("----Building network complete----")

        return x

    def conv(self, x, kernel_size, strides, filters, activation, name, pad='SAME'):
        """
        Conv ==>Batch_Norm==>Activation
        """
        #with tf.variable_scope('Net'):
        x = self.conv_layer(x, kernel_size, strides, filters, name=name, pad='SAME')
        if self.Batch_Norm:
            x = self.activation(x, activation)
        return x

    def conv_layer(self, x, kernel_size, stride, filters, name, pad='SAME'):
        x_shape = x.get_shape()
        x_channels = x_shape[3].value
        weight_shape = [kernel_size, kernel_size, x_channels, filters]
        bias_shape = [filters]
        strides = [stride, stride, stride, stride]
        weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name='weight')
        bias = tf.Variable(tf.constant(0.1, shape=bias_shape), name='biases')
        #weight = self._get_variable("weight", weight_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        #bias = self._get_variable("bias", bias_shape, initializer=tf.constant_initializer(0.0))
        y = tf.nn.conv2d(x, weight, strides=strides, padding=pad, name=name)
        if self.Batch_Norm:
            depth = filters
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            y = tf.nn.batch_normalization(y, mean, variance, shift, scale, self.EPSILON)
        y = tf.add(y, bias)
        return y

    def max_pool_layer(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding=self.MAX_PADDING, name=name)

    def activation(self, x, activation_func, name='activation_func'):
        if activation_func=='leaky':
            return tf.nn.relu(x, name='relu')
        else:
            return x

    def reorg(self, x, strides=2):
        """
        Reorg the tensor(half the size, 4* the depth)
        """
        x_shape = x.get_shape()
        B, W, H, C = x_shape[0].value, x_shape[1].value, x_shape[2].value, x_shape[3].value
        assert(W % strides == 0)
        assert(H % strides == 0)
        Ws = int(W / strides)
        Hs = int(H / strides)
        Cs = int(C * strides * strides)
        x = tf.reshape(x, shape=[-1, Ws, Hs, Cs])
        return x

    def _get_variable(self, name, shape, initializer):
        """
        创建一个函数获取变量，方便进行正则化处理等
        """
        param = 1
        for i in range(0, len(shape)):
            param *= shape[i]
        self.param_num += param

        if self.WEIGHT_DECAY > 0:
            regularizer = tf.contrib.layers.l2_regularizer(self.WEIGHT_DECAY)
        else:
            regularizer = None
        
        return tf.get_variable(name, shape=shape, regularizer=regularizer, initializer=initializer)

    def loss_layer(self, predicts, labels, scope='Loss_layer'):
        """
        Args:
            predict tensor: [batch_size, cell_size, cell_size, 19 + num_class] 19 is 9-points'-coord(18) + 1-confidence
                            last dimension: coord(18) ==> classes(num_class) ==> confidence(1)
            labels tensor:  [batch_size, cell_size, cell_size, 20 + num_class] 20 is 9-points'-coord + 1-response + 1-confidence
                            last dimension: response(1) ==> coord(18) ==> classes(num_class) ==> confidence(1)
        """
        with tf.variable_scope(scope):
            ## Predicts
            predict_coord = tf.reshape(predicts[:, :, :, :self.boundry_1], [self.Batch_Size, self.cell_size, self.cell_size, self.num_coord])
            predict_classes = tf.reshape(predicts[:, :, :, self.boundry_1:-1], [self.Batch_Size, self.cell_size, self.cell_size, self.num_class])
            predict_conf = tf.reshape(predicts[:, :, :, -1], [self.Batch_Size, self.cell_size, self.cell_size, 1])

            predict_centroids = predict_coord[:, :, :, :2*self.boxes_per_cell]
            predict_corners = predict_coord[:, :, :, 2*self.boxes_per_cell:]

            ## Ground Truth
            response = tf.reshape(labels[:, :, :, 0], [self.Batch_Size, self.cell_size, self.cell_size, 1])
            #response_for_coords = tf.tile(response, [1, 1, 1, self.num_coord * self.boxes_per_cell])  # shape: [batch, cell, cell, 18]
            labels_coord = tf.reshape(labels[:, :, :, 1:self.boundry_1+1], [self.Batch_Size, self.cell_size, self.cell_size, self.num_coord])
            labels_classes = tf.reshape(labels[:, :, :, self.boundry_1+1:], [self.Batch_Size, self.cell_size, self.cell_size, self.num_class])

            ## Offset
            #off_set = tf.constant(self.off_set, dtype=tf.float32)
            #off_set = tf.reshape(off_set, [1, self.cell_size, self.cell_size, 18 * self.boxes_per_cell])
            #off_set = tf.tile(off_set, [self.Batch_Size, 1, 1, 1])  ## off_set shape : [Batch_Size, cell_size, cell_size, 18 * boxes_per_cell]
            #off_set = tf.multiply(off_set, response_for_coords)

            #off_set_centroids = off_set[:, :, :, :2*self.boxes_per_cell]
            #off_set_corners = off_set[:, :, :, 2*self.boxes_per_cell:]

            predict_boxes_tran = tf.concat([tf.nn.sigmoid(predict_centroids), predict_corners], 3)
            ## predicts coordinates with respect to input images, [Batch_Size, cell_size, cell_size, 18]
            ## output is offset with respect to centroid, so has to add the centroid coord(top-left corners of every cell)
            ## see paper section3.2

            ## Calculate confidence (instead of IoU like in YOLOv2)
            Euclid_dist = dist(predict_boxes_tran, labels_coord)
            self.confidence = confidence_func(Euclid_dist)

            object_coef = tf.constant(self.obj_scale, dtype=tf.float32)
            noobject_coef = tf.constant(self.noobj_scale, dtype=tf.float32)
            conf_mask = tf.ones_like(response) * noobject_coef + response * object_coef # [batch. cell, cell, 1] with object:5.0, no object:0.1

            ## coordinates loss
            coord_loss = tf.losses.mean_squared_error(labels_coord, predict_boxes_tran, weights=response, scope='Coord_Loss')
            ## confidence loss, the loss between output confidence value and compute confidence
            conf_loss = tf.losses.mean_squared_error(self.confidence, predict_conf, weights=conf_mask, scope='Conf_Loss')
            ## classification loss
            class_loss = tf.losses.softmax_cross_entropy(labels_classes, predict_classes, weights=self.class_scale, scope='Class_Loss')

            tf.losses.add_loss(coord_loss)
            tf.losses.add_loss(conf_loss)
            tf.losses.add_loss(class_loss)

    def confidence_score(self, predicts, confidence):
        """
        compute the class-specific confidence scores
        see paper section 3.3
        Args:
            output tensor by net: [batch, cell_size, cell_size, 19+num_class]
        """
        predict_classes = tf.reshape(predicts[:, :, :, 18:-1], [self.Batch_Size, self.cell_size, self.cell_size, self.num_class])
        confidence = tf.tile(confidence, [1, 1, 1, self.num_class])

        class_speci_conf_score = tf.multiply(predict_classes, confidence)
        class_speci_conf_score = tf.reduce_mean(class_speci_conf_score, axis=3, keep_dims=True)
        #class_speci_conf_score = tf.nn.sigmoid(class_speci_conf_score)

        return class_speci_conf_score

    def evaluation(self):
        """
        turning network to evaluation mode, turn off Batch Norm(or Dropout)
        """
        self.is_training = False
        self.Batch_Norm = False
