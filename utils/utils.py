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

def softmax_cross_entropy(logit, label, weights):
    """
    logit: output [B, classes]
    label: ground truth [B, classes]
    weigth: [B, 1]
    """
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)

    logit = tf.exp(logit)
    logit_sum = tf.reduce_sum(logit, 1, keep_dims=True)
    logit_sum = tf.tile(logit_sum, [1, logit_shape[1]])
    softmax = tf.divide(logit, logit_sum)

    cross_entropy_loss = tf.multiply(tf.reduce_sum(-1.0 * label * tf.log(softmax), 1, keep_dims=True), weights)
    cross_entropy_loss = tf.abs(tf.reduce_sum(cross_entropy_loss))

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
    # logit_mean = tf.reduce_mean(logit, 3, keep_dims=True)
    # logit_mean = tf.tile(logit_mean, [1, 1, 1, logit_shape[3]])
    diff = tf.squared_difference(logit, label)
    diff_mean = tf.reduce_mean(diff, len(logit_shape)-1, keep_dims=True)
    error = tf.multiply(diff_mean, weights)
    error = tf.reduce_sum(error)
    return error

def confidence_func(x):
    """
    Args:
        x: a 4-D tensor: [Batch_size, cell, cell, 18]
        compute confidence score then concat to original
    Returns:
        a 4-D tensor: [Batch_Size, cell, cell, 18]
    """
    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth_in_cell_size = tf.constant(cfg.Dth, dtype=tf.float32)
    one = tf.ones_like(x, dtype=tf.float32)

    # if number in x <= dth_in_cell_size, the position in temp would be 1.0,
    # otherwise(x > dth_int_cell_size) would be 0
    temp = tf.cast(x <= dth_in_cell_size, tf.float32)

    confidence = (tf.exp(alpha * (one - x / dth_in_cell_size)) - one) / (tf.exp(alpha) - one)

    # if distance in x bigger than threshold, value calculated will be negtive,
    # use below to make the negtive to 0
    confidence = tf.multiply(confidence, temp)

    confidence = tf.reduce_mean(confidence, 3, keep_dims=True)

    return confidence

def dist(x1, x2):
    """
    Args:
        x1: 4-D tensor, [batch_size, cell, cell, 18]
        x2: 4-D tensor, [batch_size, cell, cell, 18]
    Return:
    """
    # make x1, x2 in pixel size
    x1, x2 = x1 * 32, x2 * 32
    # delta x-square, y-square, in pixel level
    diff = tf.squared_difference(x1, x2)
    # sqrt(delta x-square + delta y-square)
    predict_x = tf.stack([diff[:,:,:,0], diff[:,:,:,2], diff[:,:,:,4], diff[:,:,:,6],
                          diff[:,:,:,8], diff[:,:,:,10], diff[:,:,:,12], diff[:,:,:,14], diff[:,:,:,16]], 3)
    predict_y = tf.stack([diff[:,:,:,1], diff[:,:,:,3], diff[:,:,:,5], diff[:,:,:,7],
                          diff[:,:,:,9], diff[:,:,:,11], diff[:,:,:,13], diff[:,:,:,15], diff[:,:,:,17]], 3)

    # compute distance in pixel level
    distance = tf.sqrt(tf.add(predict_x, predict_y))

    return distance

def confidence_func9(x):
    """
    Args:
        x: a 2-D tensor: [Batch_size, 9]
        compute confidence score then concat to original
    Returns:
        a 2-D tensor: [Batch_Size, 1]
    """
    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth_in_cell_size = tf.constant(cfg.Dth, dtype=tf.float32)
    one = tf.ones_like(x, dtype=tf.float32)

    # if number in x <= dth_in_cell_size, the position in temp would be 1.0,
    # otherwise(x > dth_int_cell_size) would be 0
    temp = tf.cast(x <= dth_in_cell_size, tf.float32)

    confidence = (tf.exp(alpha * (one - x / dth_in_cell_size)) - one) / ((tf.exp(alpha) - one) + cfg.EPSILON)

    # if distance in x bigger than threshold, value calculated will be negtive,
    # use below to make the negtive to 0
    confidence = tf.multiply(confidence, temp)

    confidence = tf.reduce_mean(confidence, 1, keep_dims=True)

    return confidence

def dist9(x1, x2, pred_index, gt_index):
    """
    Args:
        x1: 2-D tensor, [batch_size, 18]
        x2: 2-D tensor, [batch_size, 18]
        pred_index: 2-D tensor, [batch_size, 2]
        gt_index  : 2-D tensor, [batch_size, 2]
    Return:
        mean confidence with shape [batch, 9]
    """
    # make x1, x2 in pixel size
    # x1, x2 = x1 * 32, x2 * 32
    # delta x**2, delta y**2, in pixel level
    shape = x1.get_shape()
    diff  = []
    for i in range(shape[0]):
        pred     = x1[i]
        gt       = x2[i]
        pred_idx = tf.cast(pred_index[i], tf.float32)
        gt_idx   = tf.cast(gt_index[i], tf.float32)
        temp     = tf.stack([pred[0]+pred_idx[0]-gt[0]-gt_idx[0], pred[1]+pred_idx[1]-gt[1]-gt_idx[1],
                             pred[2]+pred_idx[0]-gt[2]-gt_idx[0], pred[3]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[4]+pred_idx[0]-gt[2]-gt_idx[0], pred[5]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[6]+pred_idx[0]-gt[2]-gt_idx[0], pred[7]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[8]+pred_idx[0]-gt[2]-gt_idx[0], pred[9]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[10]+pred_idx[0]-gt[2]-gt_idx[0], pred[11]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[12]+pred_idx[0]-gt[2]-gt_idx[0], pred[13]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[14]+pred_idx[0]-gt[2]-gt_idx[0], pred[15]+pred_idx[1]-gt[3]-gt_idx[1],
                             pred[16]+pred_idx[0]-gt[2]-gt_idx[0], pred[17]+pred_idx[1]-gt[3]-gt_idx[1]])
        diff.append(temp)
    diff = tf.convert_to_tensor(diff)

    diff = tf.square(diff)

    predict_x = tf.stack([diff[:,0], diff[:,2], diff[:,4], diff[:,6],
                          diff[:,8], diff[:,10], diff[:,12], diff[:,14], diff[:,16]], 1)
    predict_y = tf.stack([diff[:,1], diff[:,3], diff[:,5], diff[:,7],
                          diff[:,9], diff[:,11], diff[:,13], diff[:,15], diff[:,17]], 1)

    # sqrt(delta x**2 + delta y**2)
    distance = tf.sqrt(tf.add(predict_x, predict_y))

    return distance

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
