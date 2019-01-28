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

def cross_entropy(logit, label, weights):
    """
    logit: output [B, C, C, classes]
    label: ground truth [B, C, C, classes]
    weigth: [B, C, C, 1]
    """
    logit = tf.clip_by_value(logit, 1e-10, 1.0)
    logit_shape = logit.get_shape()
    label_shape = label.get_shape()
    assert(logit_shape == label_shape)
    weight = tf.tile(weights, [1, 1, 1, logit_shape[-1]])
    
    cross_entropy_loss = tf.reduce_sum(tf.multiply(tf.multiply(label, tf.log(logit)), weight))

    return cross_entropy

def confidence_func(x):
    """
    Args:
        x: A 4-D tensor: [Batch_size, feature_size, feature_size, 18]
        compute confidence score then concat to original
    Returns:
        A 4-D tensor: [Batch_Size, feature_size, feature_size, 18]
    """
    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth_in_cell_size = tf.constant(cfg.Dth, dtype=tf.float32)
    param1 = tf.ones_like(x, dtype=tf.float32)

    # if number in x <= dth_in_cell_size, the position in temp would be 0,
    # otherwise would be 1
    temp = tf.cast(x <= dth_in_cell_size, tf.float32)

    confidence = (tf.exp(alpha * (param1 - x / dth_in_cell_size)) - param1) / (tf.exp(alpha) - param1)

    # if distance in x bigger than threshold, value calculated will be negtive,
    # use below to make the negtive to 0
    confidence = tf.multiply(confidence, temp)

    confidence = tf.reduce_mean(confidence, 3, keep_dims=True)
    return confidence

def dist(x1, x2):
    """
    Args:
        x1: 4-D tensor, [batch_size, cell_size, cell_size, 18]
        x2: 4-D tensor, [batch_size, cell_size, cell_size, 18]
    Return:
    """
    # delta x, y
    diff = tf.abs((x1 - x2))
    # delta x-square, y-square, in pixel level
    diff = tf.square(diff) * 32
    # sqrt(delta x-square + delta y-square)
    predict_x = tf.stack([diff[:, :, :, 0], diff[:, :, :, 2], diff[:, :, :, 4], diff[:, :, :, 6],
                            diff[:, :, :, 8], diff[:, :, :, 10], diff[:, :, :, 12], diff[:, :, :, 14], diff[:, :, :, 16]], 3)
    predict_y = tf.stack([diff[:, :, :, 1], diff[:, :, :, 3], diff[:, :, :, 5], diff[:, :, :, 7],
                            diff[:, :, :, 9], diff[:, :, :, 11], diff[:, :, :, 13], diff[:, :, :, 15], diff[:, :, :, 17]], 3)

    # compute distance in pixel level
    distance = tf.sqrt(tf.add(predict_x, predict_y))
    shape = distance.get_shape()

    return distance


def confidence_thresh(cscs, predicts, threshold=cfg.CONF_THRESHOLD):
    """
    decide in predicts which to be pruned by threshold through cscs.
    Args:
        cscs: class-specific confidence score  with shape: [cell_size, cell_size, 1]
        predicts: output coord tensor(convert to numpy)  with shape: [cell_size, cell_size, 18]
        threshold: conf_thresh, defined in config.py
    Return:
        output: a numpy feature tensor with shape: [cell_size, cell_size, 18]
    """
    out_tensor = np.ones_like(cscs, dtype=np.float32)  # initialize out_tensor
    out_tensor[cscs <= threshold] = 0
    out_tensor = np.tile(out_tensor, [1, 1, predicts.shape[2]])  # expand out_tensor to the shape of predicts
    out_tensor = np.multiply(out_tensor, predicts) # multiply element-wise, when outtensor number is 0 means it will be pruned
    return out_tensor

def nms(input_tensor, cscs):
    """
    get the maximum confidence value to response
    Args:
        input_tensor: output_tensor from confidence_thresh with shape: [cell_size, cell_size, 18]
        cscs: class-specific confidence score with shape: [cell_size, cell_size, 1]
    Return:
        a numpy feature tensor with shape: [cell_size, cell_size, 18]
    """
    res = np.zeros_like(cscs, dtype=np.float32)
    max_idx = np.argmax(cscs)
    col, row = divmod(max_idx, 13)
    row -= 1
    res[col, row, :] = 1
    res = np.tile(res, [1, 1, input_tensor.shape[2]])
    output = np.multiply(res, input_tensor)
    return output

def nms33(input_tensor, cscs):
    """
    get the maximum confidence score tensor from confidence_thresh
    Args:
        cscs: class-specific confidence score with shape: [cell_size, cell_size, 1]
        input_tensor: out_tensor from confidence_thresh with shape: [cell_size, cell_size, 18]
    Return:
        output: a numpy feature tensor with shape: [cell_size, cell_size, 18]
    """
    res = np.zeros_like(cscs, dtype=np.float32)
    for i in range(1, input_tensor.shape[0]-1, 2):
        for j in range(1, input_tensor.shape[1]-1, 2):
            temp = cscs[i-1:i+2, j-1:j+2, :]
            temp_max = np.argmax(temp)
            k, l, __ = np.where(temp == temp_max)
            #k, l = k[0], l[0]
            res[k+i-1, l+j-1, :] = 1
    res = np.tile(res, [1, 1, input_tensor.shape[2]])
    out_tensor = np.multiply(res, input_tensor)
    return out_tensor


def get_region_boxes(output, num_classes):
    """
    Return a list which elements are coordinates of all boxes
    Args:
        output: 3-D tensor with shape [cell, cell, 19+num_classes]
        num_classes:
    Return:
        A list
    """
    anchor_num = 1

    h, w, d = output.shape[0], output.shape[1], output.shape[2]
    output_coord = output[:, :, :18]
    output_cls   = output[:, :, 18:-1]
    output_confs = output[:, :, -1].reshape(h, w, 1)

    output_coord = np.concatenate([sigmoid_func(output_coord[:, :, :2]), output_coord[:, :, 2:]], 2)
    output_confs = sigmoid_func(output_confs)
    output_cls = softmax(output_cls)

    boxes = []
    for i in range(h):
        for j in range(w):
            max_conf = -1
            if output_confs[i][j][0] == 0:
                continue
            else:
                xc = output_coord[i][j][0] + j
                yc = output_coord[i][j][1] + i
                x1 = output_coord[i][j][2] + j
                y1 = output_coord[i][j][3] + i
                x2 = output_coord[i][j][4] + j
                y2 = output_coord[i][j][5] + i
                x3 = output_coord[i][j][6] + j
                y3 = output_coord[i][j][7] + i
                x4 = output_coord[i][j][8] + j
                y4 = output_coord[i][j][9] + i
                x5 = output_coord[i][j][10] + j
                y5 = output_coord[i][j][11] + i
                x6 = output_coord[i][j][12] + j
                y6 = output_coord[i][j][13] + i
                x7 = output_coord[i][j][14] + j
                y7 = output_coord[i][j][15] + i
                x8 = output_coord[i][j][16] + j
                y8 = output_coord[i][j][17] + i
                output_conf = output_confs[i][j][0]
                if max_conf < output_conf:
                    max_conf = output_conf
                    max_idi, max_idj = i, j
                cls_max_conf = np.max(output_cls[i][j])
                cls_max_id, = np.where(output_cls[i][j] == cls_max_conf)
                cls_max_id = cls_max_id[0]
            box = [xc/13, yc/13, x1/13, y1/13, x2/13, y2/13, x3/13, y3/13, x4/13, y4/13, x5/13, y5/13, x6/13, y6/13, x7/13, y7/13, x8/13, y8/13,
                    output_conf, cls_max_conf, cls_max_id]
            boxes.append(box)

    return boxes

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


def corner_confidence9(gt_corners, pr_corners, th=80, sharpness=2, im_width=640, im_height=480):
    ''' gt_corners: Ground-truth 2D projections of the 3D bounding box corners, shape: (18,) type: list
        pr_corners: Prediction for the 2D projections of the 3D bounding box corners, shape: (18,), type: list
        th        : distance threshold, type: int
        sharpness : sharpness of the exponential that assigns a confidence value to the distance
        -----------
        return    : a list of shape (9,) with 9 confidence values
    '''
    dist = np.subtract(gt_corners, pr_corners)
    dist = dist.reshape(9, 2)
    dist[:, 0] = dist[:, 0] * im_width
    dist[:, 1] = dist[:, 1] * im_height

    eps = 1e-5
    dist  = np.sqrt(np.sum((dist)**2, axis=1))
    mask  = (dist < th)
    conf  = np.exp(sharpness * (1.0 - dist/th)) - 1
    conf0 = np.exp(np.array([sharpness])) - 1 + eps
    conf  = conf / conf0.repeat(18).reshape(9,1)
    # conf = 1.0 - dist/th
    conf  = mask * conf
    return np.mean(conf)

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
