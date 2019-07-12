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

import config as cfg


def sigmoid_func(x, derivative=False):
    return x*(1.0-x) if derivative else 1.0/(1.0+np.exp(-x))

def softmax(X, theta = 1.0, axis = None):
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

def softmax_cross_entropy(logit, label, weights):
    """
    logit: output [B, classes]
    label: ground truth [B, classes]
    weigth: [B, 1]
    """
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)
    epsilon = tf.constant(cfg.EPSILON, dtype=tf.float32)

    logit     = tf.exp(logit)
    logit_sum = tf.reduce_sum(logit, 1, keep_dims=True)
    logit_sum = tf.tile(logit_sum, (1, logit_shape[1]))
    softmax_t = logit / (logit_sum + epsilon)
    weights   = tf.tile(weights, (1, label_shape[1]))

    cross_entropy_loss = tf.reduce_sum(-1.0 * label * tf.log(softmax_t), 1, keep_dims=True) * weights
    cross_entropy_loss = tf.reduce_sum(cross_entropy_loss)

    return cross_entropy_loss

def conf_mean_squared_error(logit, label, weights):
    """
    logit: output conf map  [batch, cell, cell, 1]
    label: ground truth conf map [batch, cell, cell, 1]
    weights: coef [batch, cell, cell, 1]
    """
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)

    diff = tf.squared_difference(logit, label)
    assert(diff.get_shape() == logit_shape)
    error = tf.multiply(diff, weights)
    error = tf.reduce_sum(error)
    return error

def coord_mean_squared_error(logit, label, weights):
    """
    logit: output coords  [batch, 18]
    label: ground truth coords [batch, 18]
    weights: coef [batch, 1]
    """
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)

    diff = tf.squared_difference(logit, label)
    diff_mean = tf.reduce_mean(diff, 1, keep_dims=True)
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
    pred_x_shape = pred_x.get_shape()
    pred_y_shape = pred_y.get_shape()
    gt_x_shape   = gt_x.get_shape()
    gt_y_shape   = gt_y.get_shape()
    assert(pred_x_shape[1:] == gt_x_shape[1:])
    assert(pred_y_shape[1:] == gt_y_shape[1:])

    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth = tf.constant(cfg.Dth, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    epsilon = tf.constant(cfg.EPSILON, dtype=tf.float32)

    pred_x = pred_x / 13.0 * 640
    pred_y = pred_y / 13.0 * 480
    gt_x   = gt_x   / 13.0 * 640
    gt_y   = gt_y   / 13.0 * 480
    dist_x = tf.squared_difference(pred_x, gt_x)
    dist_y = tf.squared_difference(pred_y, gt_y)
    dist   = tf.sqrt(dist_x + dist_y)

    # if number in x <= dth_in_cell_size, the position in temp would be 1.0,
    # otherwise(x > dth_int_cell_size) would be 0
    temp = tf.cast(dist < dth, tf.float32)

    confidence = (tf.exp(alpha * (one - dist / dth)) - one) / (tf.exp(alpha) - one + epsilon)

    confidence = confidence * temp

    confidence = tf.reduce_mean(confidence, 3, keep_dims=True)

    return confidence

def get_max_index(confidence):
    """
    confidence: 2-D tensor [cell_size, cell_size]
    return the index of maximum value of confidence
    """
    assert(confidence.get_shape()[0]==13)
    assert(confidence.get_shape()[1]==13)
    max_val  = tf.reduce_max(confidence)
    bool_idx = tf.equal(confidence, max_val)
    int_idx  = tf.where(bool_idx)
    maxi = int_idx[0, 0]
    maxj = int_idx[0, 1]
    return maxi, maxj

def corner_confidences9(gt_corners, pr_corners, th=80, sharpness=2.0, im_width=640, im_height=480):
    """
    gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18 x 169)
    pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18 x 169)
    th        : distance threshold, type: int
    sharpness : sharpness of the exponential that assigns a confidence value to the distance
    -----------
    return    : a torch.FloatTensor of shape (nA,) with 9 confidence values
    """
    shape = gt_corners.get_shape()
    nA = shape[1]
    sharpness = tf.constant(sharpness, dtype=tf.float32)
    one = tf.constant(1.0, dtype=tf.float32)
    eps = tf.constant(cfg.EPSILON, dtype=tf.float32)
    distthresh = tf.constant(th, dtype=tf.float32)

    dist = gt_corners - pr_corners
    dist = tf.reshape(tf.transpose(dist, (1, 0)), (nA, 9, 2))
    dist_x = dist[:, :, 0] * im_height # (nA, 9), in image size
    dist_y = dist[:, :, 1] * im_width
    dist = tf.transpose(tf.stack([dist_x, dist_y]), (1, 2, 0))

    dist = tf.sqrt(tf.reduce_sum(tf.square(dist), 2))  # nA X 9
    mask = tf.cast(dist < th, tf.float32)
    conf = tf.exp(sharpness * (one - dist/distthresh)) - one
    conf0 = tf.exp(sharpness * (one - tf.zeros_like(dist))) - one + eps
    conf = conf / conf0
    conf = mask * conf  # nA X 9
    mean_conf = tf.reduce_mean(conf, 1) # (nA,)
    return mean_conf

def corner_confidence9(gt_corners, pr_corners, th=80, sharpness=2.0, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: tensor
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: tensor
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values
    '''
    dist = gt_corners - pr_corners  # (18,)
    image_size = tf.Variable([im_width, im_height], dtype=tf.float32, trainable=False)
    image_size = tf.reshape(image_size, (1, 2))
    image_size = tf.tile(image_size, (9,1))
    dist = tf.reshape(dist, (9,2))
    dist = dist * image_size  # 9 X 2

    eps = tf.constant(cfg.EPSILON)
    sharpness = tf.constant(sharpness)
    one = tf.constant(1.0)
    th = tf.constant(th, dtype=tf.float32)

    dist  = tf.sqrt(tf.reduce_sum(tf.square(dist), 1))  # [9,]
    mask  = tf.cast(dist < th, tf.float32)
    conf  = tf.exp(sharpness * (one - dist/th)) - one
    conf0 = tf.exp(sharpness * (one - tf.zeros_like(dist))) - one + eps
    conf  = conf / conf0
    conf  = mask * conf
    return tf.reduce_mean(conf)

def get_predict_boxes(output, num_classes):
    h, w, _ = output.shape[0], output.shape[1], output.shape[2]
    output_coord = output[:, :, :18]
    output_coord = np.concatenate([sigmoid_func(output_coord[:, :, :2]), output_coord[:, :, 2:]], 2)
    # output_cls   = output[:, :, 18:-1]
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
