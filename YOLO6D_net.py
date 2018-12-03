import tensorflow as tf
import numpy as np

"""
This is the script to try to reproduce YOLO-6D
input: a mini-batch of images
train:

"""

class YOLO6D:
    WEIGHT_DECAY = 0.0001
    MAX_PADDING = 'SAME'
    EPSILON = 1e-10
    learning_rate = None
    optimizer = None
    loss = None

    def __init__(self, input_size, name, activation_func=tf.nn.relu):
        """
        placeholder定义输入、lr，
        """
        self.param_num = 0
        sess = tf.Session()
        layers = []
        x = tf.placeholder(dtype=tf.float32, shape=input_size, name='input')
        #y = tf.placeholder(dtype=tf.float32, shape=)

    def conv_layer(self, x, name, kernel_size, strides, filters, pad='SAME'):
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

#    def _loss():


    def get_optimizer(self):
        ##choose an optimizer to train the network
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

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