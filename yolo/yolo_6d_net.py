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


    def __init__(self, is_training=True):
        """
        Input images: [416 * 416 * 3]
        output tensor: [13 * 13 * (18 + 1 + num_classes)]
            self.input_images ==> self.logit
        Input labels: [batch * 13 * 13 * 20 + num_classes]
        """
        self.Batch_Size = cfg.BATCH_SIZE
        self.WEIGHT_DECAY = cfg.WEIGHT_DECAY
        self.MAX_PADDING = cfg.MAX_PAD
        self.EPSILON = cfg.EPSILON
        self.learning_rate = cfg.LEARNING_RATE
        self.optimizer = None
        self.total_loss = None
        self.disp = cfg.DISP
        self.param_num = 0
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.image_size = cfg.IMAGE_SIZE

        self.num_class = cfg.NUM_CLASSES
        self.Batch_Norm = cfg.BATCH_NORM
        self.ALPHA = cfg.ALPHA
        self.cell_size = cfg.CELL_SIZE
        self.num_coord = cfg.NUM_COORD  ## 18: 9 points, 8 corners + 1 centroid

        self.obj_scale = cfg.CONF_OBJ_SCALE
        self.noobj_scale = cfg.CONF_NOOBJ_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

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
        self.Euclid_dist = None

        if is_training:
            # self.total_loss = self.loss_layer(self.logit, self.labels)
            self.loss_layer(self.logit, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('Total loss', self.total_loss)

        #self.conf_value = tf.reshape(self.logit[:, :, :, -1], [-1, self.cell_size, self.cell_size, 1])
        # self.conf_score = self.confidence_score(self.logit, self.confidence)


    def _build_net(self, input):
        if self.disp:
            print("\n--------------Building network---------------")
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
            print("----------Building network complete----------\n")

        return x

# ======================== Net definition ==================================

    def conv(self, x, kernel_size, strides, filters, activation, name, pad='SAME'):
        """
        Conv ==> Batch_Norm ==> Bias
        """
        x_shape = x.get_shape()
        x_channels = x_shape[3].value
        weight_shape = [kernel_size, kernel_size, x_channels, filters]
        bias_shape = [filters]
        stride = [strides, strides, strides, strides]
        weight = tf.Variable(tf.truncated_normal(weight_shape, stddev=0.1), name='weight') / (self.cell_size * self.cell_size)
        bias = tf.Variable(tf.constant(0.1, shape=bias_shape), name='biases') / (self.cell_size * self.cell_size)

        x = tf.nn.conv2d(x, weight, strides=stride, padding=pad, name=name)
        if self.Batch_Norm:
            depth = filters
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            x = tf.nn.batch_normalization(x, mean, variance, shift, scale, self.EPSILON)

        x = tf.add(x, bias)

        if self.Batch_Norm:
            x = self.activation(x, activation)

        return x

    def max_pool_layer(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding=self.MAX_PADDING, name=name)

    def activation(self, x, activation_func, name='activation_func'):
        if activation_func=='leaky':
            return tf.nn.leaky_relu(x, alpha=0.1, name='leaky')
        elif activation_func=='relu':
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

# ======================= Net definition end ===============================

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
            predict_conf = tf.reshape(predicts[:, :, :, -1], [self.Batch_Size, self.cell_size, self.cell_size, 1])  # get predicted confidence
            # conf_mask = tf.zeros([self.Batch_Size, self.cell_size, self.cell_size, 1], dtype=tf.int32)
            pred_tensor = []  # restore tensors
            pred_index  = []  # restore index

            # get the max confidence tensor
            for i in range(self.Batch_Size):
                pred_conf = predict_conf[i]
                pred_conf = tf.reshape(pred_conf, [self.cell_size, self.cell_size])
                pred_i, pred_j = get_max_index(pred_conf)
                temp_tensor = predicts[i, pred_i, pred_j, :]
                # value_i, value_j = tf.cast(max_index_i, tf.float32), tf.cast(max_index_j, tf.float32)
                # temp_tensor[0]  = tf.add(temp_tensor[0], value_i)
                # temp_tensor[1]  += value_j
                # temp_tensor[2]  += value_i
                # temp_tensor[3]  += value_j
                # temp_tensor[4]  += value_i
                # temp_tensor[5]  += value_j
                # temp_tensor[6]  += value_i
                # temp_tensor[7]  += value_j
                # temp_tensor[8]  += value_i
                # temp_tensor[9]  += value_j
                # temp_tensor[10] += value_i
                # temp_tensor[11] += value_j
                # temp_tensor[12] += value_i
                # temp_tensor[13] += value_j
                # temp_tensor[14] += value_i
                # temp_tensor[15] += value_j
                # temp_tensor[16] += value_i
                # temp_tensor[17] += value_j
                pred_tensor.append(temp_tensor)
                pred_index.append([pred_i, pred_j])
                conf_mask = tf.sparse_tensor_to_dense(tf.SparseTensor([[i, pred_i, pred_j, 0]], [1.0], [self.Batch_Size, self.cell_size, self.cell_size, 1]))
            pred_tensor = tf.convert_to_tensor(pred_tensor)
            pred_index  = tf.convert_to_tensor(pred_index)

            predict_centroids = pred_tensor[:, :2*self.boxes_per_cell]
            predict_corners   = pred_tensor[:, 2*self.boxes_per_cell:self.boundry_1]
            predict_coord_tr  = tf.concat([tf.nn.sigmoid(predict_centroids), predict_corners], 1)
            predict_classes   = pred_tensor[:, self.boundry_1:-1]
            # predict_boxes_tr  = tf.concat([tf.nn.sigmoid(predicts[:,:,:,:2]), predicts[:,:,:,2:self.boundry_1]], 3)


            ## Ground Truth
            response = tf.reshape(labels[:, :, :, 0], [self.Batch_Size, self.cell_size, self.cell_size, 1])
            gt_tensor = []
            gt_index = []

            # get the responsible tensor
            for i in range(self.Batch_Size):
                gt_resp = response[i]
                gt_resp = tf.reshape(gt_resp, [self.cell_size, self.cell_size])
                gt_i, gt_j = get_max_index(gt_resp)
                temp_tensor = labels[i, gt_i, gt_j, :]
                # value_i, value_j = tf.cast(gt_i, tf.float32), tf.cast(gt_j, tf.float32)
                # temp_tensor[1]  += value_i
                # temp_tensor[2]  += value_j
                # temp_tensor[3]  += value_i
                # temp_tensor[4]  += value_j
                # temp_tensor[5]  += value_i
                # temp_tensor[6]  += value_j
                # temp_tensor[7]  += value_i
                # temp_tensor[8]  += value_j
                # temp_tensor[9]  += value_i
                # temp_tensor[10] += value_j
                # temp_tensor[11] += value_i
                # temp_tensor[12] += value_j
                # temp_tensor[13] += value_i
                # temp_tensor[14] += value_j
                # temp_tensor[15] += value_i
                # temp_tensor[16] += value_j
                # temp_tensor[17] += value_i
                # temp_tensor[18] += value_j
                gt_tensor.append(temp_tensor)
                gt_index.append([gt_i, gt_j])
            gt_tensor = tf.convert_to_tensor(gt_tensor)
            gt_index  = tf.convert_to_tensor(gt_index)

            labels_coord   = gt_tensor[:, 1:self.boundry_1+1]
            labels_classes = gt_tensor[:, self.boundry_1+1: ]
            labels_conf    = []

            ## Calculate confidence (instead of IoU like in YOLOv2)
            dist = dist9(predict_coord_tr, labels_coord, pred_index, gt_index)
            confidence = confidence_func9(dist) # [batch, 1]

            for i in range(self.Batch_Size):
                labels_conf.append(response[i] * confidence[i,0])
            labels_conf = tf.convert_to_tensor(labels_conf)
            print(labels_conf.get_shape())


            ## Set coefs for loss
            object_coef   = tf.constant(self.obj_scale, dtype=tf.float32)
            noobject_coef = tf.constant(self.noobj_scale, dtype=tf.float32)

            conf_coef     = tf.add(tf.ones_like(response)*noobject_coef, conf_mask*object_coef) # [batch. cell, cell, 1] with object:5.0, no object:0.1
            coord_coef    = tf.ones([self.Batch_Size, 1]) * self.coord_scale
            class_coef    = tf.ones([self.Batch_Size, 1]) * self.class_scale


            ## Compute losses
            # conf_loss = tf.losses.mean_squared_error(self.confidence, predict_conf, weights=conf_coef, scope='Conf_Loss')
            conf_loss = mean_squared_error(predict_conf, labels_conf, weights=conf_coef)

            # coord_loss = tf.losses.mean_squared_error(labels_coord, predict_boxes_valid, weights=self.coord_scale, scope='Coord_Loss')
            coord_loss = mean_squared_error(predict_coord_tr, labels_coord, weights=coord_coef)

            class_loss = softmax_cross_entropy(labels_classes, predict_classes, weights=class_coef)

            # loss = conf_loss + coord_loss + class_loss
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

    def evaluation_off(self):
        self.is_training = True
        self.Batch_Norm = True

