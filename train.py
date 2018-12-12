from __future__ import print_function
import sys
import argparse
import YOLO6D_net
import datetime
import os
from utils import *
from MeshPly import MeshPly
import config as cfg
from YOLO6D_net import YOLO6D_net
import tensorflow as tf
import numpy as np


class Solver(object):
    def __init__(self, net, data):
        self.net = net
        self.data = data
        self.batch_size = cfg.BATCH_SIZE
        self.weight_file = cfg.WEIGHTS_FILE
        self.max_iter = cfg.MAX_ITER
        self.inital_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEP
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.variable_to_restore = tf.global_variables()
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)
        self.ckpt_file = os.path.join(self.output_dir, 'save.ckpt')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.get_variable(
            'global_step', [], initializer=tf.constant_initializer(0.0), trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.inital_learning_rate, self.global_step, 
                                                        self.decay_steps, self.decay_rate, self.staircase, name='learning_rate')
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
            self.net.total_loss, global_step=self.global_step)
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        gpu_option = tf.GPUOptions()
        config = tf.ConfigProto(gpu_option=gpu_option)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer)
        if self.weight_file is not None:
            print('Restoring weights from: ' + self.weight_file)
            self.restorer.restore(self.sess, self.weight_file)

        self.writer.add_graph(self.sess.graph)

    #def train(self, epoch):

def update_config_paths(data_dir, weights_file):
    cfg.DATA_DIR = data_dir
    cfg.DATASETS_DIR = os.path.join(data_dir, 'datasets')
    cfg.CACHE_DIR = os.path.join(cfg.DATASETS_DIR, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.DATASETS_DIR, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.DATASETS_DIR, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_6D.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu
    
    if args.data_dir != cfg.DATA_DIR:
        update_config_paths(args.data_dir, args.weights)
        
    os.environ['CUDA_VISABLE_DEVICES'] = cfg.GPU

    yolo = YOLO6D_net()
    #datasets = 

    #solver = Solver(yolo, datasets)
    print("------start training------")
    #solver.train()
    print("-------training end-------")

if __name__ == "__main__":
    
    main()