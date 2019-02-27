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

class YOLO6D_net:


    def __init__(self, is_training=True):
        """
        Input images: [batch, 416 * 416 * 3]
        Input labels: [batch * 13 * 13 * (19 + num_classes)]
        output tensor: [batch, 13 * 13 * (19 + num_classes)]
        """
        self.is_training = is_training
        self.Batch_Size = cfg.BATCH_SIZE
        self.EPSILON = cfg.EPSILON
        self.learning_rate = cfg.LEARNING_RATE
        self.total_loss = None
        self.disp = cfg.DISP
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.image_size = cfg.IMAGE_SIZE

        self.num_class = cfg.NUM_CLASSES
        self.Batch_Norm = cfg.BATCH_NORM
        self.cell_size = cfg.CELL_SIZE

        self.obj_scale = cfg.CONF_OBJ_SCALE
        self.noobj_scale = cfg.CONF_NOOBJ_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE

        self.boundry_1 = 9 * 2   ## Seperate coordinates
        self.boundry_2 = self.num_class

        self.input_images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')

        self.logit = self.build_networks(self.input_images)

        if self.is_training:
            self.gt_conf = None
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 19 + self.num_class], name='labels')
            self.total_loss = self.loss_layer(self.logit, self.labels)
            # self.loss_layer(self.logit, self.labels)
            # self.total_loss = tf.losses.get_total_loss()
            tf.summary.tensor_summary('Total loss', self.total_loss)

# ======================== Net definition ==================================

    def build_networks(self, inputs):
        if self.disp:
            print("\n--------------Building network---------------")
        net = self.conv_layer(inputs, [3, 3, 3, 32], name = '0_conv')

        net = self.pooling_layer(net, name = '1_pool')
        net = self.conv_layer(net, [3, 3, 32, 64], name = '2_conv')

        net = self.pooling_layer(net, name = '3_pool')
        net = self.conv_layer(net, [3, 3, 64, 128], name = '4_conv')
        net = self.conv_layer(net, [1, 1, 128, 64], name = '5_conv')
        net = self.conv_layer(net, [3, 3, 64, 128], name = '6_conv')

        net = self.pooling_layer(net, name = '7_pool')
        net = self.conv_layer(net, [3, 3, 128, 256], name = '8_conv')
        net = self.conv_layer(net, [1, 1, 256, 128], name = '9_conv')
        net = self.conv_layer(net, [3, 3, 128, 256], name = '10_conv')

        net = self.pooling_layer(net, name = '11_pool')
        net = self.conv_layer(net, [3, 3, 256, 512], name = '12_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], name = '13_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], name = '14_conv')
        net = self.conv_layer(net, [1, 1, 512, 256], name = '15_conv')
        net16 = self.conv_layer(net, [3, 3, 256, 512], name = '16_conv')

        net = self.pooling_layer(net16, name = '17_pool')
        net = self.conv_layer(net, [3, 3, 512, 1024], name = '18_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512], name = '19_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], name = '20_conv')
        net = self.conv_layer(net, [1, 1, 1024, 512], name = '21_conv')
        net = self.conv_layer(net, [3, 3, 512, 1024], name = '22_conv')
        net = self.conv_layer(net, [3, 3, 1024, 1024], name = '23_conv')
        net24 = self.conv_layer(net, [3, 3, 1024, 1024], name = '24_conv')

        net = self.conv_layer(net16, [1, 1, 512, 64], name = '26_conv')
        net = self.reorg(net)

        net = tf.concat([net, net24], 3)

        net = self.conv_layer(net, [3, 3, int(net.get_shape()[3]), 1024], name = '29_conv')
        net = self.conv_layer(net, [1, 1, 1024, 19 + self.num_class], batch_norm=False, name = '30_conv', activation='linear')

        if self.disp:
            print("----------Building network complete----------\n")
        return net

    def conv_layer(self, inputs, shape, batch_norm = True, name = '0_conv', activation = 'leaky'):
        initializer = tf.contrib.layers.xavier_initializer()
        weight = tf.Variable(initializer(shape), name='weight')
        # weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        biases = tf.Variable(tf.constant(1.0, shape=[shape[3]]), name='biases')

        conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)

        if batch_norm:
            depth = shape[3]
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            conv = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
            # conv = tf.add(conv, biases)
        else:
            conv = tf.add(conv, biases)

        if activation == 'leaky':
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activation == 'relu':
            conv = tf.nn.relu(conv)
        elif activation == 'linear':
            return conv

        return conv

    def pooling_layer(self, inputs, name = '1_pool'):
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

    def loss_layer(self, predicts, labels, scope='Loss_layer'):
        """
        Args:
            predict tensor: [batch_size, cell_size, cell_size, 19 + num_class] 19 is 9-points'-coord(18) + 1-confidence
                            last dimension: coord(18) ==> classes(num_class) ==> confidence(1)
            labels tensor:  [batch_size, cell_size, cell_size, 20 + num_class] 20 is 9-points'-coord + 1-response + 1-confidence
                            last dimension: response(1) ==> coord(18) ==> classes(num_class) ==> confidence(1)
        """
        with tf.variable_scope(scope):

            ## Ground Truth
            response = tf.reshape(labels[:, :, :, 0], [self.Batch_Size, self.cell_size, self.cell_size, 1])

            gt_tensor = []
            gt_idx = []
            # get the responsible tensor's index
            for i in range(self.Batch_Size):
                gt_resp = tf.reshape(response[i], [self.cell_size, self.cell_size])
                gt_i, gt_j = get_max_index(gt_resp)
                temp_tensor = labels[i, gt_i, gt_j,:]  # shape: [32,]
                gt_tensor.append(temp_tensor)
                gt_idx.append([gt_i, gt_j])
            gt_tensor = tf.convert_to_tensor(gt_tensor)  # shape: [batch, 32], store object tensors
            gt_idx    = tf.convert_to_tensor(gt_idx)  # shape: [batch, 2]
            #metric
            labels_coord   = gt_tensor[:, 1:self.boundry_1+1]  # for later coord loss
            labels_classes = gt_tensor[:, self.boundry_1+1: ]  # for later class loss

            gt_coords = labels[:, :, :, 1:self.boundry_1+1]  # [batch, cell, cell, 18]
            ground_true_boxes_x = tf.transpose(tf.stack([gt_coords[:,:,:,0], gt_coords[:,:,:,2], gt_coords[:,:,:,4], gt_coords[:,:,:,6],
                                            gt_coords[:,:,:,8], gt_coords[:,:,:,10], gt_coords[:,:,:,12], gt_coords[:,:,:,14], gt_coords[:,:,:,16]]),
                                            (1, 2, 3, 0))  # [Batch, cell, cell, 9], for later conf calculate
            ground_true_boxes_y = tf.transpose(tf.stack([gt_coords[:,:,:,1], gt_coords[:,:,:,3], gt_coords[:,:,:,5], gt_coords[:,:,:,7],
                                            gt_coords[:,:,:,9], gt_coords[:,:,:,11], gt_coords[:,:,:,13], gt_coords[:,:,:,15], gt_coords[:,:,:,17]]),
                                            (1, 2, 3, 0))  # [Batch, cell, cell, 9], for later conf calculate


            ## Predicts
            predict_conf      = tf.reshape(predicts[:, :, :, 0], [self.Batch_Size, self.cell_size, self.cell_size, 1])  # get predicted confidence
            predict_boxes_tr  = tf.concat([tf.nn.sigmoid(predicts[:,:,:,1:3]), predicts[:,:,:,3:self.boundry_1+1]], 3)
            # offset for predicts
            off_set_x  = np.tile(np.reshape(np.array([np.arange(13)] * 13 ), (13, 13, 1)), (1, 1, 9))
            off_set_y  = np.transpose(off_set_x, (1, 0, 2))
            off_set_x  = np.tile(np.transpose(np.reshape(off_set_x, (13, 13, 9, 1)), (3, 0, 1, 2)), (self.Batch_Size, 1, 1, 1))  # [Batch, cell, cell, 9]
            off_set_y  = np.tile(np.transpose(np.reshape(off_set_y, (13, 13, 9, 1)), (3, 0, 1, 2)), (self.Batch_Size, 1, 1, 1))  # [Batch, cell, cell, 9]
            predict__x = tf.transpose(tf.stack([predict_boxes_tr[:,:,:,0], predict_boxes_tr[:,:,:,2], predict_boxes_tr[:,:,:,4],
                                                predict_boxes_tr[:,:,:,6], predict_boxes_tr[:,:,:,8], predict_boxes_tr[:,:,:,10],
                                                predict_boxes_tr[:,:,:,12], predict_boxes_tr[:,:,:,14], predict_boxes_tr[:,:,:,16]]),
                                                (1,2,3,0))  # [Batch, cell, cell, 9]
            predict__y = tf.transpose(tf.stack([predict_boxes_tr[:,:,:,1], predict_boxes_tr[:,:,:,3], predict_boxes_tr[:,:,:,5],
                                                predict_boxes_tr[:,:,:,7], predict_boxes_tr[:,:,:,9], predict_boxes_tr[:,:,:,11],
                                                predict_boxes_tr[:,:,:,13], predict_boxes_tr[:,:,:,15], predict_boxes_tr[:,:,:,17]]),
                                                (1,2,3,0))  # [Batch, cell, cell, 9]
            pred_box_x = predict__x + off_set_x  # predict boxes x coordinates with offset, for later conf calculate
            pred_box_y = predict__y + off_set_y  # predict boxes y coordinates with offset, for later conf calculate
            pred_boxes = tf.transpose(tf.stack([pred_box_x[:,:,:,0], pred_box_y[:,:,:,0],
                                                pred_box_x[:,:,:,1], pred_box_y[:,:,:,1],
                                                pred_box_x[:,:,:,2], pred_box_y[:,:,:,2],
                                                pred_box_x[:,:,:,3], pred_box_y[:,:,:,3],
                                                pred_box_x[:,:,:,4], pred_box_y[:,:,:,4],
                                                pred_box_x[:,:,:,5], pred_box_y[:,:,:,5],
                                                pred_box_x[:,:,:,6], pred_box_y[:,:,:,6],
                                                pred_box_x[:,:,:,7], pred_box_y[:,:,:,7],
                                                pred_box_x[:,:,:,8], pred_box_y[:,:,:,8]]), (1,2,3,0))  # predict coords [batch, cell, cell, 18]
            pred_boxes = tf.concat([pred_boxes, predicts[:,:,:,19:]], 3)  # [batch, cell, cell, 31], without confidence

            pred_tensor = []  # restore tensor
            # get the max confidence tensor and its index
            for i in range(self.Batch_Size):
                pred_conf = predict_conf[i]
                pred_conf = tf.reshape(pred_conf, [self.cell_size, self.cell_size])
                if self.obj_scale == 0.0:
                    # means in pre train
                    pred_i, pred_j = gt_idx[i, 0], gt_idx[i, 1]
                else:
                    # in training
                    pred_i, pred_j = get_max_index(pred_conf)
                temp_tensor = pred_boxes[i, pred_i, pred_j, :]
                pred_tensor.append(temp_tensor)
            pred_tensor = tf.convert_to_tensor(pred_tensor)  # shape: [batch, 31], store tensors with max_confidence
            # metric
            predict_coord_tr  = pred_tensor[:, :self.boundry_1]  # for later coord loss
            predict_classes   = pred_tensor[:, self.boundry_1:]  # for later class loss


            ## Calculate confidence (instead of IoU like in YOLOv2)
            labels_conf = confidence9(pred_box_x, pred_box_y, ground_true_boxes_x, ground_true_boxes_y) # [batch, cell, cell, 1]
            self.gt_conf = labels_conf

            ## Set coefs for loss
            object_coef   = tf.constant(self.obj_scale, dtype=tf.float32)
            noobject_coef = tf.constant(self.noobj_scale, dtype=tf.float32)

            conf_coef     = tf.add(tf.ones_like(response)*noobject_coef, response*object_coef) # [batch, cell, cell, 1] with object:5.0, no object:0.1
            coord_coef    = tf.ones([self.Batch_Size, 1]) * self.coord_scale # [batch, 1]
            class_coef    = tf.ones([self.Batch_Size, 1]) * self.class_scale # [batch, 1]


            ## Compute losses
            conf_loss = conf_mean_squared_error(predict_conf, labels_conf, weights=conf_coef)

            coord_loss = coord_mean_squared_error(predict_coord_tr, labels_coord, weights=coord_coef)

            class_loss = softmax_cross_entropy(predict_classes, labels_classes, weights=class_coef)
            # class_loss = tf.losses.softmax_cross_entropy(labels_classes, predict_classes)

            loss = conf_loss + coord_loss + class_loss

            total_loss = []
            total_loss.append(loss)
            total_loss.append(conf_loss)
            total_loss.append(coord_loss)
            total_loss.append(class_loss)
            return total_loss


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
