from __future__ import print_function
import sys
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



if __name__ == "__main__":
    
    #Training settings
    datacfg = sys.argv[1]
    weight_file = sys.argv[2]

    ##Parse configuration files
    data_options  = read_data_cfg(datacfg)
    trainlist     = data_options['train']
    testlist      = data_options['valid']
    nsamples      = file_lines(trainlist)
    gpus          = data_options['gpus']  # e.g. 0,1,2,3
    gpus = '0'
    meshname      = data_options['mesh']
    num_workers   = int(data_options['num_workers'])
    backupdir     = data_options['backup']
    diam          = float(data_options['diam'])
    vx_threshold  = diam * 0.1
    if not os.path.exists(backupdir):
        makedirs(backupdir)

    ##Training Parameters
    max_epoch = 700  ## max_batch * batch_size / nSamples + 1
    use_cuda = True
    eps = 1e-5
    save_interval = 10 #epoches
    dot_interval = 70 #batches
    best_acc = -1

    ##Testing Parameters
    conf_thresh = 0.1
    nms_thresh = 0.4
    iou_thresh = 0.5
    img_width = 640
    img_height = 480

    # Specify the model and loss
    Yolo = YOLO6D_net()
    region_loss = Yolo.total_loss
    
    ##Variables to save
    training_iters     = []
    training_losses    = []
    testing_iters      = []
    testing_losses     = []
    testing_err_trans  = []
    testing_err_pixel  = []
    testing_err_angle  = []
    testing_accuracies = []

    ##Get the intrinsic camera matrix, mesh, vertices, and corners of the model
    mesh = MeshPly(meshname)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].T
    corners3D = get_3D_corners(vertices)
    internal_calibration = get_camera_intrinsic()

    # Specify the number of workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if use_cuda else {}


    