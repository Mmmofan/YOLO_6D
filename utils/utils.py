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

def confidence_func(x):
    """
    Args:
        A 4-D tensor: [Batch_size, feature_size, feature_size, 18]
        compute confidence score then concat to original
    Returns:
        A 4-D tensor: [Batch_Size, feature_size, feature_size, 18]
    """
    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth_in_cell_size = tf.constant(cfg.Dth / cfg.CELL_SIZE, dtype=tf.float32)
    param1 = tf.ones_like(x, dtype=tf.float32)
    confidence = (tf.exp(alpha * (param1 - x / dth_in_cell_size))) / (tf.exp(alpha) - param1)
    confidence = tf.reduce_mean(confidence, 3, keep_dims=True)
    return confidence

def dist(x1, x2):
    """
    Args:
        x1: 4-D tensor, [batch_size, cell_size, cell_size, 18]
        x2: 4-D tensor, [batch_size, cell_size, cell_size, 18]
    Return:
    """
    predict_x = tf.stack([x1[:, :, :, 0], x1[:, :, :, 2], x1[:, :, :, 4], x1[:, :, :, 6],  
                            x1[:, :, :, 8], x1[:, :, :, 10], x1[:, :, :, 12],  x1[:, :, :, 14], x1[:, :, :, 16]], 3)
    predict_y = tf.stack([x1[:, :, :, 1], x1[:, :, :, 3], x1[:, :, :, 5], x1[:, :, :, 7],  
                            x1[:, :, :, 9], x1[:, :, :, 11], x1[:, :, :, 13],  x1[:, :, :, 15], x1[:, :, :, 17]], 3)
    gt_x = tf.stack([x2[:, :, :, 0], x2[:, :, :, 2], x2[:, :, :, 4], x2[:, :, :, 6],  
                        x2[:, :, :, 8], x2[:, :, :, 10], x2[:, :, :, 12], x2[:, :, :, 14], x2[:, :, :, 16]], 3)
    gt_y = tf.stack([x2[:, :, :, 1], x2[:, :, :, 3], x2[:, :, :, 5], x2[:, :, :, 7],  
                        x2[:,:,:,9], x2[:,:,:,11], x2[:,:,:,13], x2[:,:,:,15], x2[:,:,:,17]], 3)
    distance = tf.sqrt(tf.add(tf.square(predict_x - gt_x), tf.square(predict_y - gt_y)))
    ### dist: 4-D tensor [batch_size, cell_size, cell_size, 9]
    return distance

def preprocess_image(images, image_size=(416, 416)):
    """
    Preprocess images to make the type, size and dimensions correct
    """
    # copy the images
    image_cp = np.copy(images).astype(np.float32)
    #resize images
    image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, image_size)
    #normalize
    image_normalize = image_resized.astype(np.float32) / 255.0
    #add one dimension (batch)
    image_expanded = np.expand_dims(image_normalize, axis=0)

    return image_expanded

def postprocess(ouput_tensor, image_shape=(416, 416), threshold=cfg.CONF_THRESHOLD):
    """
    Args:
        output_tensor: computed by net of shape [batch, cell_size, cell_size, 19+num_class]
        trans to Numpy array first!
    Returns:
        tensors which have coords in real images
    """
    coordinates = ouput_tensor[:, :, :, :18]
    class_prob = ouput_tensor[:, :, :, 18:-1]
    confidence = ouput_tensor[:, :, :, -1]

    # Restore the coordinates fit real images
    off_set = np.transpose(np.reshape(np.array(
                                [np.arange(cfg.CELL_SIZE)] * cfg.CELL_SIZE * 18 * cfg.BOXES_PER_CELL),
                                (18, cfg.CELL_SIZE, cfg.CELL_SIZE)),
                                (1, 2, 0)).astype(np.float32)
    off_set = np.reshape(off_set, [1, cfg.CELL_SIZE, cfg.CELL_SIZE, 18 * cfg.BOXES_PER_CELL])
    off_set = np.tile(off_set, [cfg.BATCH_SIZE, 1, 1, 1])
    ## off_set shape : [Batch_Size, cell_size, cell_size, 18 * boxes_per_cell]
    off_set_centroids = off_set[:, :, :, :2]
    off_set_corners = off_set[:, :, :, 2:]
    predict_boxes_tran = np.concatenate([sigmoid_func(coordinates[:, :, :, :2]) + off_set_centroids,
                                        coordinates[:, :, :, 2:] + off_set_corners], 3)
    predict_boxes = np.multiply(predict_boxes_tran, float(cfg.CELL_SIZE))  ## Coordinates in real images

    # Cut the box, assert the boxes' bounding less than 416
    predict_boxes[predict_boxes > 416] = 416
    predict_boxes[predict_boxes < 0] = 0


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
            k, l, __ = np.where(temp == temp_max)  # k, l, __ is a tuple
            #k, l = k[0], l[0]
            res[k+i-1, l+j-1, :] = 1
    res = np.tile(res, [1, 1, input_tensor.shape[2]])
    out_tensor = np.multiply(res, input_tensor)
    return out_tensor

def compute_average(orig_tensor, cscs, out_tensor):
    """
    Unfinish
    """
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
    
    output_coord = output[:, :, :18]
    output_cls   = output[:, :, 18:-1]
    output_confs = output[:, :, -1]
    h, w, d = output.shape[0], output.shape[1], output.shape[2]
    off_set = np.transpose(np.reshape(np.array([np.arange(h)] * w * 18 * anchor_num), (18, h, w)), (1, 2, 0))
    off_set[output_confs == 0] = 0
    off_set_centroids = off_set[:, :, :2]
    off_set_corners = off_set[:, :, 2:]

    output_coord = np.concatenate([(sigmoid_func(output_coord[:, :, :2]) + off_set_centroids), (output_coord[:, :, 2:] + off_set_corners)], 2)
    output_confs = sigmoid_func(output_confs)
    output_cls = softmax(output_cls)

    boxes = []
    for i in range(h):
        for j in range(w):
            max_conf = -1
            if output_confs[i][j] == 0:
                continue
            else:
                xc = output_coord[i][j][0]
                yc = output_coord[i][j][1]
                x1 = output_coord[i][j][2]
                y1 = output_coord[i][j][3]
                x2 = output_coord[i][j][4]
                y2 = output_coord[i][j][5]
                x3 = output_coord[i][j][6]
                y3 = output_coord[i][j][7]
                x4 = output_coord[i][j][8]
                y4 = output_coord[i][j][9]
                x5 = output_coord[i][j][10]
                y5 = output_coord[i][j][11]
                x6 = output_coord[i][j][12]
                y6 = output_coord[i][j][13]
                x7 = output_coord[i][j][14]
                y7 = output_coord[i][j][15]
                x8 = output_coord[i][j][16]
                y8 = output_coord[i][j][17]
                output_conf = output_confs[i][j]
                if max_conf < output_conf:
                    max_conf = output_conf
                    max_idi, max_idj = i, j
                cls_max_conf = np.max(output_cls[i][j])
                cls_max_id, = np.where(output_cls[i][j] == cls_max_conf)
                cls_max_id = cls_max_id[0]
            box = [xc/h, yc/w, x1/h, y1/w, x2/h, y2/w, x3/h, y3/w, x4/h, y4/w, x5/h, y5/w, x6/h, y6/w, x7/h, y7/w, x8/h, y8/w, 
                    output_conf, cls_max_conf, cls_max_id]
            boxes.append(box)

    return boxes

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
