# -*- coding: utf-8 -*-
# ---------------------
# utils for yolo6d
# @Author: Fan, Mo
# @Email: fmo@nullmax.ai
# ---------------------

import os

import cv2
import numpy as np
import tensorflow as tf

import yolo.config as cfg


def sigmoid_func(x, derivative=False):
    """
    Compute sigmoid of x element-wise
    """
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)
    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    # multiply y against the theta parameter,
    y = y * float(theta)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()
    return p

def softmax_cross_entropy(label, logit, weights):
    """
    logit: output [B, classes]
    label: ground truth [B, classes]
    weigth: [B, 1]
    """
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)
    epsilon = tf.constant(cfg.EPSILON, dtype=tf.float32)

    logit = tf.exp(logit)
    logit_sum = tf.reduce_sum(logit, 1, keep_dims=True)
    logit_sum = tf.tile(logit_sum, (1, logit_shape[1])) + epsilon
    softmax = tf.divide(logit, logit_sum)
    weights = tf.tile(weights, (1, label_shape[1]))

    cross_entropy_loss = tf.multiply(tf.reduce_sum(-1.0 * label * tf.log(softmax), 1, keep_dims=True), weights)
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss)

    return cross_entropy_loss

def mean_squared_error(logit, label, weights):
    """
    logit: output
    label: ground truth
    weights: coef
    """
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)
    diff = tf.squared_difference(logit, label)
    diff_mean = tf.reduce_mean(diff, len(logit_shape)-1, keep_dims=True)
    error = tf.multiply(diff_mean, weights)
    error = tf.reduce_sum(error)
    return error

def confidence9(pred_x, pred_y, gt_x, gt_y):
    """
    Args:
        pred_x: 4-D tensor, [batch_size, cell_size, cell_size, 9]
        pred_y: 4-D tensor, [batch_size, cell_size, cell_size, 9]
        gt_x  : 4-D tensor, [batch_size, cell_size, cell_size, 9]
        gt_y  : 4-D tensor, [batch_size, cell_size, cell_size, 9]
    Return:
        confidence: [batch, cell_size, cell_size, 1]
    """
    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth_in_cell_size = tf.constant(cfg.Dth, dtype=tf.float32)
    one = tf.ones_like(pred_x, dtype=tf.float32)

    pred_x = pred_x * 32
    pred_y = pred_y * 32
    gt_x   = gt_x   * 32
    gt_y   = gt_y   * 32
    dist_x = tf.square(pred_x - gt_x)
    dist_y = tf.square(pred_y - gt_y)
    dist   = tf.sqrt(dist_x + dist_y)

    # if number in x <= dth_in_cell_size, the position in temp would be 1.0,
    # otherwise(x > dth_int_cell_size) would be 0
    temp = tf.cast(dist <= dth_in_cell_size, tf.float32)

    confidence = (tf.exp(alpha * (one - dist / dth_in_cell_size)) - one) / (tf.exp(alpha) - one + cfg.EPSILON)

    # if distance in x bigger than threshold, value calculated will be negtive,
    # use below to make the negtive to 0
    confidence = tf.multiply(confidence, temp)

    confidence = tf.reduce_mean(confidence, 3, keep_dims=True)

    return confidence

def get_max_index(confidence):
    """
    confidence: 2-D tensor [cell_size, cell_size]
    return the index of maximum value of confidence
    """
    max_val  = tf.reduce_max(confidence)
    bool_idx = tf.equal(confidence, max_val)
    int_idx  = tf.where(bool_idx)
    maxi = int_idx[0][0]
    maxj = int_idx[0][1]
    return maxi, maxj

def get_predict_boxes(output, num_classes):
    h, w, _ = output.shape[0], output.shape[1], output.shape[2]
    output_coord = output[:, :, :18]
    output_coord = np.concatenate([sigmoid_func(output_coord[:, :, :2]), output_coord[:, :, 2:]], 2)
    output_cls   = output[:, :, 18:-1]
    output_conf  = output[:, :, -1]

    max_conf = np.max(output_conf)
    max_conf_id = np.where(output_conf == max_conf)
    idi, idj = max_conf_id[0][0], max_conf_id[1][0]

    xc = output_coord[idi][idj][0] + idj
    yc = output_coord[idi][idj][1] + idi
    x1 = output_coord[idi][idj][2] + idj
    y1 = output_coord[idi][idj][3] + idi
    x2 = output_coord[idi][idj][4] + idj
    y2 = output_coord[idi][idj][5] + idi
    x3 = output_coord[idi][idj][6] + idj
    y3 = output_coord[idi][idj][7] + idi
    x4 = output_coord[idi][idj][8] + idj
    y4 = output_coord[idi][idj][9] + idi
    x5 = output_coord[idi][idj][10] + idj
    y5 = output_coord[idi][idj][11] + idi
    x6 = output_coord[idi][idj][12] + idj
    y6 = output_coord[idi][idj][13] + idi
    x7 = output_coord[idi][idj][14] + idj
    y7 = output_coord[idi][idj][15] + idi
    x8 = output_coord[idi][idj][16] + idj
    y8 = output_coord[idi][idj][17] + idi

    box = [xc/13, yc/13, x1/13, y1/13, x2/13, y2/13, x3/13, y3/13, x4/13, y4/13,
           x5/13, y5/13, x6/13, y6/13, x7/13, y7/13, x8/13, y8/13]
    return box

def calcAngularDistance(gt_rot, pr_rot):
    rotDiff = np.dot(gt_rot, np.transpose(pr_rot))
    trace = np.trace(rotDiff)
    return np.rad2deg(np.arccos((trace-1.0)/2.0))

def compute_transformation(points_3D, transformation):
    return transformation.dot(points_3D)

def compute_projection(points_3D, transformation, internal_calibration):
    projections_2d = np.zeros((2, points_3D.shape[1]), dtype='float32')
    camera_projection = (internal_calibration.dot(transformation)).dot(points_3D)
    projections_2d[0, :] = camera_projection[0, :]/camera_projection[2, :]
    projections_2d[1, :] = camera_projection[1, :]/camera_projection[2, :]
    return projections_2d

def pnp(points_3D, points_2D, cameraMatrix):
    """
    Use PnP algorithm compute 6D pose
    """
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8,1), dtype='float32')

    assert points_2D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'

    _, R_exp, t = cv2.solvePnP(points_3D,
                                # points_2D,
                                np.ascontiguousarray(points_2D[:, :2]).reshape((-1,1,2)),
                                cameraMatrix,
                                distCoeffs)
    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t

###############################################################################

def get_all_files(directory):
    files = []

    for f in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, f)):
            files.append(os.path.join(directory, f))
        else:
            files.extend(get_all_files(os.path.join(directory, f)))
    return files

def read_truths(lab_path):
    if os.path.getsize(lab_path):
        truths = np.loadtxt(lab_path)
        truths = truths.reshape(truths.size/21, 21) # to avoid single truth problem
        return truths
    else:
        return np.array([])

def read_truths_args(lab_path, min_box_scale):
    truths = read_truths(lab_path)
    new_truths = []
    for i in range(truths.shape[0]):
        new_truths.append([truths[i][0], truths[i][1], truths[i][2], truths[i][3], truths[i][4],
            truths[i][5], truths[i][6], truths[i][7], truths[i][8], truths[i][9], truths[i][10],
            truths[i][11], truths[i][12], truths[i][13], truths[i][14], truths[i][15], truths[i][16], truths[i][17], truths[i][18]])
    return np.array(new_truths)

def file_lines(thefilepath):
    count = 0
    thefile = open(thefilepath, 'rb')
    while True:
        buffer = thefile.read(8192*1024)
        if not buffer:
            break
        count += buffer.count('\n')
    thefile.close( )
    return count

def read_data_cfg(datacfg):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(datacfg, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        line = line.strip()
        if line == '':
            continue
        key,value = line.split('=')
        key = key.strip()
        value = value.strip()
        options[key] = value
    return options

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_3D_corners(vertices):

    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1,8)) ), axis=0)
    return corners

def get_camera_intrinsic():
    K = np.zeros((3, 3), dtype='float64')
    K[0, 0], K[0, 2] = 572.4114, 325.2611
    K[1, 1], K[1, 2] = 573.5704, 242.0489
    K[2, 2] = 1.
    return K
