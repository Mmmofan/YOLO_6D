import tensorflow as tf
import numpy as np
from utils import *
import sys

"""
This is the script to try to reproduce YOLO-6D
input: a mini-batch of images
train:

"""

class YOLO6D:

    Batch_Size = 45
    WEIGHT_DECAY = 0.0001
    MAX_PADDING = 'SAME'
    EPSILON = 1e-10
    learning_rate = None
    optimizer = None
    loss = None
    disp = True
    param_num = 0

    num_class = 0
    Batch_Norm = False
    ALPHA = 2.0
    cell_size = 13
    num_coord = 18  ## 9 points, 8 corners + 1 centroid

    obj_scale = 5.0
    noobje_scale = 0.1
    conf_scale = 1.0
    coord_scale = 1.0

    def __init__(self):
        """
        placeholder定义输入
        """
        self.image_size = 416
        self.input_images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='Input')
        self.logit = self._build_net(self.input_images)
        self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 19 + self.num_class], name='Labels')
        self.loss = self.loss_layer(self.logit, self.labels)
        self.total_loss = tf.losses.get_total_loss()


    def _build_net(self, input_size):
        if self.disp:
            print("-----building network-----")
        self.x = self.conv(self.input_images, 3, 1, 32, num=1)
        self.x = self.max_pool_layer(self.x, name='MaxPool1')
        self.x = self.conv(self.x, 3, 1, 64, num=2)
        self.x = self.max_pool_layer(self.x, name='MaxPool2')
        self.x = self.conv(self.x, 3, 1, 128, num=3)
        self.x = self.conv(self.x, 1, 1, 64, num=4)
        self.x = self.conv(self.x, 3, 1, 128, num=5)
        self.x = self.max_pool_layer(self.x, name='MaxPool3')
        self.x = self.conv(self.x, 3, 1, 256, num=6)
        self.x = self.conv(self.x, 1, 1, 128, num=7)
        self.x = self.conv(self.x, 3, 1, 256, num=8)
        self.x = self.max_pool_layer(self.x, name='MaxPool4')
        self.x = self.conv(self.x, 3, 1, 512, num=9)
        self.x = self.conv(self.x, 1, 1, 256, num=10)
        self.x = self.conv(self.x, 3, 1, 512, num=11)
        self.x = self.conv(self.x, 1, 1, 256, num=12)
        self.x = self.conv(self.x, 3, 1, 512, num=13)
        self.x_ps = self.conv(self.x, 1, 1, 64, num=14)    #add a pass through layer
        self.x_ps = self.conv(self.x_ps, 3, 2, 256, num=15)   ##
        self.x = self.max_pool_layer(self.x, name='MaxPool5')    #continue straight layer
        self.x = self.conv(self.x, 3, 1, 1024, num=16)
        self.x = self.conv(self.x, 1, 1, 512, num=17)
        self.x = self.conv(self.x, 3, 1, 1024, num=18)
        self.x = self.conv(self.x, 1, 1, 512, num=19)
        self.x = self.conv(self.x, 3, 1, 1024, num=20)
        self.x = self.conv(self.x, 3, 1, 1024, num=21)
        self.x = self.conv(self.x, 3, 1, 1024, num=22)
        self.x = self.merge_layer(self.x, self.x_ps, name='Merge')
        self.x = self.conv(self.x, 3, 1, 1024, num=23)
        self.x = self.conv(self.x, 1, 1, 19 + self.num_class - 1, num=24)
        return self.x

    def conv(self, x, kernel_size, strides, filters, num, pad='SAME'):
        """
        Conv ==> ReLU ==> Batch_Norm
        """
        x = self.conv_layer(x, kernel_size, strides, filters, pad='SAME', name='Conv:{0}'.format(num))
        x = self.activation(x)
        if self.Batch_Norm:
            x = self.bn(x)
        return x

    def conv_layer(self, x, kernel_size, strides, filters, name, pad='SAME'):
        x_shape = x.get_shape()
        x_channels = x_shape[3].value
        weight_shape = [kernel_size, kernel_size, x_channels, filters]
        bias_shape = [filters]
        weight = self._get_variable(name, weight_shape, initializer=tf.truncated_normal_initializer)
        bias = self._get_variable(name, bias_shape, initializer=tf.constant(0.0))
        y = tf.nn.conv2d(x, weight, strides=strides, padding=pad, name=name)
        y = tf.add(y, bias, name=name)
        return y

    def merge_layer(self, x1, x2, name):
        """
        input are 2 tensors from different conv_layer
        """
        x_list = [x1, x2]
        y = tf.concat(3, x_list, name=name)
        return y

    def max_pool_layer(self, x, name):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding=self.MAX_PADDING, name=name)
 
    def bn(self, x, name='BN'):
        axes = [d for d in range(len(x.get_shape()))]
        gamma = self._get_variable('gamma', [], initializer=tf.constant_initializer(1.0))
        beta  = self._get_variable('beta', [], initializer=tf.constant_initializer(0.0))
        x_mean, x_variance = tf.nn.moments(x, axes)
        y = tf.nn.batch_normalization(x, x_mean, x_variance, beta, gamma, self.EPSILON, name=name)
        return y
 
    def activation(self, x, name='activation_func'):
        return tf.nn.relu(x, name='relu')

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
        
        return tf.get_variable(name, shape = shape, regularizer=regularizer, initializer=initializer)

    def get_optimizer(self):
        ##choose an optimizer to train the network
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def loss_layer(self, predicts, labels, scope='Loss_layer'):
        self.predict_coord = tf.reshape(predicts[:, :, :, :18], [self.Batch_Size, self.cell_size, self.cell_size, self.num_coord])
        self.predict_classes = tf.reshape(predicts[:, :, :, 18:], [self.Batch_Size, self.cell_size, self.cell_size, self.num_class])
    
        self.labels_coord = tf.reshape(labels[:, :, :, 18], [self.Batch_Size, self.cell_size, self.cell_size, self.num_coord])
        self.labels_classes = tf.reshape(labels[:, :, :, 18:-1], [self.Batch_Size, self.cell_size, self.cell_size, self.num_class])
        self.labels_conf = tf.reshape(labels[:, :, :, -1], [self.Batch_Size, self.cell_size, self.cell_size, 1])

        self.dt_x = dist(self.predict_coord, self.labels_coord)
        self.predict_conf = confidence_func(self.dt_x)

        ## coord loss
        #self.        