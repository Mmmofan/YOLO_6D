#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------
# solver file for yolo-6d
# @Author: Fan, Mo
# @Email: fmo@nullmax.ai
# ---------------------

from __future__ import print_function

import argparse
import datetime
import time
import os

import numpy as np
import tensorflow as tf

import yolo.config as cfg
from linemod import Linemod
from utils.MeshPly import MeshPly
from utils.timer import Timer
from utils.utils import *
from yolo.yolo_6d_net import YOLO6D_net


class Solver(object):

    def __init__(self, net, data, arg=None):

        #Set parameters for training and testing
        self.meshname = data.meshname
        self.backupdir = data.backupdir
        self.vx_threshold = data.vx_threshold

        self.mesh = MeshPly(self.meshname)
        self.vertices = np.c_[np.array(self.mesh.vertices), np.ones((len(self.mesh.vertices), 1))].T
        self.corners3D = get_3D_corners(self.vertices)
        self.internal_calibration = get_camera_intrinsic()
        self.best_acc = -1
        self.testing_errors_trans = []
        self.testing_errors_angle = []
        self.testing_errors_pixel = []
        self.testing_accuracies = []

        self.net = net
        self.data = data
        if arg.batch == 0:
            self.batch_size = cfg.BATCH_SIZE
        else:
            self.batch_size = arg.batch
        self.epoch = cfg.EPOCH
        self.weight_file = os.path.join(cfg.WEIGHTS_DIR, arg.weights)
        self.cache_file  = cfg.CACHE_FILE
        self.max_iter = int(len(data.imgname) / self.batch_size)
        self.inital_learning_rate = cfg.LEARNING_RATE  # 0.0001
        self.decay_steps = cfg.DECAY_STEP
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = cfg.OUTPUT_DIR

        if arg.pre:
            self.variable_to_restore = tf.global_variables()[:-2]
        else:
            self.variable_to_restore = tf.global_variables()[:-10]

        self.variable_to_save = tf.global_variables()
        self.restorer = tf.train.Saver(self.variable_to_restore, max_to_keep=3)
        self.saver = tf.train.Saver(self.variable_to_save, max_to_keep=3)
        self.cacher = tf.train.Saver(self.variable_to_save, max_to_keep=3)
        self.ckpt_file = os.path.join(self.weight_file, arg.weights)
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        if arg.pre:
            boundaries = cfg.PRE_BOUNDARIES
        else:
            boundaries = cfg.BOUNDARIES
        learning_rate = cfg.L_R_STAIR
        self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, learning_rate, name='learning_rate')
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            self.net.total_loss[0], global_step=self.global_step)
        # self.learning_rate = tf.train.exponential_decay(self.inital_learning_rate, self.global_step, self.decay_steps,
                                                        # self.decay_rate, self.staircase, name='learning_rate')
        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
            # self.net.total_loss[0], global_step=self.global_step)

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.global_step.initializer)
        trainable = tf.trainable_variables()
        for i in range(len(trainable)-8):
            self.sess.run(trainable[i].initializer)

        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.averages_op = self.ema.apply(tf.trainable_variables())
        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        if self.weight_file is not None:
            print('\n----------Restoring weights from: {}------batch: {}--'.format(self.weight_file, self.batch_size))
            self.restorer.restore(self.sess, self.weight_file)
        self.writer.add_graph(self.sess.graph)


    def train(self):
        self.net.evaluation_off()
        train_timer = Timer()
        load_timer = Timer()

        epoch = 0
        best_loss = 1e8
        while epoch <= self.epoch:
            for step in range(1, self.max_iter-1):
                load_timer.tic()
                images, gt_label, labels = self.data.next_batches()
                load_timer.toc()

                feed_dict = {self.net.input_images: images, self.net.target: gt_label, self.net.labels: labels}

                if step % self.summary_iter == 0:
                    if step % (self.summary_iter * 4) == 0:
                        train_timer.tic()
                        summary_str, loss, _ = self.sess.run(
                            [self.summary_op, self.net.total_loss, self.train_op],
                            feed_dict=feed_dict
                        )
                        train_timer.toc()

                        log_str = ('\n   {}, Epoch:{}, Step:{}, Learning rate:{},\n'
                                   '   Loss: {:5.3f}, conf_loss: {:5.3f}, coord_loss: {:5.3f}, class_loss: {:5.3f},\n'
                                   '   Speed: {:.3f}s/iter, Load: {:.3f}s/iter, Remain: {}').format(
                            datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                            epoch,
                            int(step),
                            round(self.learning_rate.eval(session=self.sess), 6),
                            loss[0], loss[1], loss[2], loss[3],
                            train_timer.average_time,
                            load_timer.average_time,
                            train_timer.remain(step, self.max_iter))
                        print("=======================================================================")
                        print(log_str)

                        if loss[0] < best_loss:
                            print('best loss!')
                            self.cacher.save(self.sess, self.cache_file)
                            best_loss = loss[0]
                        if loss[0] is None:
                            break

                        # test
                        # self.test()

                        # if self.testing_accuracies[-1] > self.best_acc:
                            # self.best_acc = self.testing_accuracies[-1]
                            # print('   best model so far!')
                            # print('   Save weights to %s/yolo_6d.ckpt' % (self.output_dir))
                            # self.saver.save(self.sess, '%s/yolo_6d.ckpt' % (self.output_dir), global_step=self.global_step)

                    else:
                        train_timer.tic()
                        summary_str, _ = self.sess.run(
                            [self.summary_op, self.train_op],
                            feed_dict=feed_dict)
                        train_timer.toc()

                    self.writer.add_summary(summary_str, step)

                else:
                    train_timer.tic()
                    self.sess.run(self.train_op, feed_dict=feed_dict)
                    train_timer.toc()

                if step % self.save_iter == 0:
                    datetime.datetime.now().strftime('%m/%d %H:%M:%S')
                    print('   Save checkpoint file to: {}'.format(
                        self.weight_file))
                    print("=======================================================================")
                    self.saver.save(self.sess, self.weight_file,
                                    global_step=self.global_step)
            epoch += 1
            self.data.batch = 0

        print('\n   Save final checkpoint file to: {}'.format(self.weight_file))
        self.saver.save(self.sess, self.weight_file, global_step=self.global_step)


    def test(self):
        # turn off batch norm
        self.net.evaluation()
        test_timer = Timer()
        load_timer = Timer()
        im_width = 640
        im_height = 480
        eps = 1e-5

        load_timer.tic()
        images, labels = self.data.next_batches_test()
        truths = self.data.get_truths() #2-D [Batch, params]
        load_timer.toc()

        feed_dict = {self.net.input_images: images, self.net.labels: labels}
        #predicts: [batch, cell, cell, coords + classes + confidence]
        predicts = self.sess.run(self.net.logit, feed_dict=feed_dict)  # run
        #confidence = predicts[:, :, :, -1]
        testing_error_trans = 0.0
        testing_error_angle = 0.0
        testing_error_pixel = 0.0
        testing_samples = 0.0
        errs_2d = []
        errs_3d = []
        errs_trans = []
        errs_angle = []
        errs_corner2D = []

        #all_boxes = []
        #Iterate throught test examples
        for batch_idx in range(cfg.BATCH_SIZE):
            test_timer.tic()
            #conf_sco = confidence_score[batch_idx]
            logit = predicts[batch_idx] # 3-D
            logit = logit * 10.0
            truth = truths[batch_idx]
            #num_gts = truth[0]

            # prune tensors with low confidence (< 0.1)
            #logit = confidence_thresh(conf_sco, pred)

            # get the maximum of 3x3 neighborhood
            #logit_nms = nms33(logit, conf_sco)
            #logit = nms(logit, conf_sco)

            # compute weighted average of 3x3 neighborhood
            #logit = compute_average(predicts[batch_idx], conf_sco, logit_nms)

            # get all the boxes coordinates
            # 1st: ground true boxes
            box_gt = [truth[1], truth[2], truth[3], truth[4], truth[5],
                      truth[6], truth[7], truth[8], truth[9], truth[10],
                      truth[11], truth[12], truth[13], truth[14], truth[15],
                      truth[16], truth[17], truth[18], 1.0, 1.0, truth[0]]

            # 2nd: predict boxes
            box_pr = get_predict_boxes(logit, cfg.NUM_CLASSES)

            #denomalize the corner prediction
            corners2D_gt = np.array(np.reshape(box_gt[:18], [9, 2]), dtype='float32')
            corners2D_pr = np.array(np.reshape(box_pr[:18], [9, 2]), dtype='float32')
            corners2D_gt[:, 0] = corners2D_gt[:, 0] * im_width
            corners2D_gt[:, 1] = corners2D_gt[:, 1] * im_height
            corners2D_pr[:, 0] = corners2D_pr[:, 0] * im_width
            corners2D_pr[:, 1] = corners2D_pr[:, 1] * im_height

            # Compute corner prediction error
            corner_norm = np.linalg.norm(corners2D_gt - corners2D_pr, axis=1)
            corner_dist = np.mean(corner_norm)
            errs_corner2D.append(corner_dist)

            # Compute [R|t] by pnp
            R_gt, t_gt = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),
                                corners2D_gt, np.array(self.internal_calibration, dtype='float32'))
            R_pr, t_pr = pnp(np.array(np.transpose(np.concatenate((np.zeros((3, 1)), self.corners3D[:3, :]), axis=1)), dtype='float32'),
                                corners2D_pr, np.array(self.internal_calibration, dtype='float32'))

            # Compute errors

            # Compute translation error
            trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
            errs_trans.append(trans_dist)

            # Compute angle error
            angle_dist   = calcAngularDistance(R_gt, R_pr)
            errs_angle.append(angle_dist)

            # Compute pixel error
            Rt_gt        = np.concatenate((R_gt, t_gt), axis=1)
            Rt_pr        = np.concatenate((R_pr, t_pr), axis=1)
            proj_2d_gt   = compute_projection(self.vertices, Rt_gt, self.internal_calibration)
            proj_2d_pred = compute_projection(self.vertices, Rt_pr, self.internal_calibration)
            norm         = np.linalg.norm(proj_2d_gt - proj_2d_pred, axis=0)
            pixel_dist   = np.mean(norm)
            errs_2d.append(pixel_dist)

            # Compute 3D distances
            transform_3d_gt   = compute_transformation(self.vertices, Rt_gt)
            transform_3d_pred = compute_transformation(self.vertices, Rt_pr)
            norm3d            = np.linalg.norm(transform_3d_gt - transform_3d_pred, axis=0)
            vertex_dist       = np.mean(norm3d)
            errs_3d.append(vertex_dist)

            # Sum errors
            testing_error_trans  += trans_dist
            testing_error_angle  += angle_dist
            testing_error_pixel  += pixel_dist
            testing_samples      += 1
        test_timer.toc()
        # Compute 2D projection, 6D pose and 5cm5degree scores
        px_threshold = 5
        acc = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
        acc3d = len(np.where(np.array(errs_3d) <= self.vx_threshold)[0]) * 100. / (len(errs_3d)+eps)
        acc5cm5deg = len(np.where((np.array(errs_trans) <= 0.05) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
        mean_corner_err_2d = np.mean(errs_corner2D)
        nts = float(testing_samples)
        # Print test statistics
        print("   Mean corner error is %f" % (mean_corner_err_2d))
        print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
        print('   Acc using {} vx 3D Transformation = {:.2f}%'.format(self.vx_threshold, acc3d))
        print('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
        print('   Translation error: %f, angle error: %f' % (testing_error_trans/(nts+eps), testing_error_angle/(nts+eps)) )

        # Register losses and errors for saving later on
        self.testing_errors_trans.append(testing_error_trans/(nts+eps))
        self.testing_errors_angle.append(testing_error_angle/(nts+eps))
        self.testing_errors_pixel.append(testing_error_pixel/(nts+eps))
        self.testing_accuracies.append(acc)
        test_timer.average_time
        load_timer.average_time

    def __del__(self):
        self.sess.close()


def update_config_paths(data_dir, weights_file):
    cfg.DATA_DIR     = data_dir
    cfg.CACHE_DIR    = os.path.join(cfg.DATA_DIR, 'cache')
    cfg.OUTPUT_DIR   = os.path.join(cfg.DATA_DIR, 'output')
    cfg.WEIGHTS_DIR  = os.path.join(cfg.DATA_DIR, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datacfg', default='cfg/ape.data', type=str)
    parser.add_argument('--pre', default=False, type=bool)
    parser.add_argument('--gpu', default='2', type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--weights', default="yolo_6d.ckpt", type=str)
    parser.add_argument('--batch', default=0, type=int)
    args = parser.parse_args()

    if len(args.datacfg) == 0:
        print('No datacfg file specified')
        return

    if args.pre:
        print("Pre-training... ")
        cfg.CONF_OBJ_SCALE = 0.0
        cfg.CONF_NOOBJ_SCALE = 0.0

    if args.data_dir != cfg.DATA_DIR:
        update_config_paths(args.data_dir, args.weights)

    gpu_device = '/gpu:' + args.gpu
    # os.environ['CUDA_VISABLE_DEVICES'] = args.gpu

    with tf.device(gpu_device):
        yolo = YOLO6D_net()
        datasets = Linemod('train', arg=args.datacfg)
        solver = Solver(yolo, datasets, arg=args)

        print("\n-----------------------------start training----------------------------")
        tic = time.clock()
        solver.train()
        toc = time.clock()
    print("All training time: {}h".format((toc - tic) / 3600.0))
    print("------------------------------training end-----------------------------\n")

if __name__ == "__main__":

    main()
