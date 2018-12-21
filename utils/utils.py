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
    """
    Compute sigmoid of x element-wise
    """
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

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
        cscs: class-specific confidence score  [batch_size, cell_size, cell_size, 1]
        predicts: output coord tensor(convert to numpy)  [batch_size, cell_size, cell_size, 18]
        threshold: conf_thresh, defined in config.py
    Return:
        output: a numpy feature tensor  [batch_size, cell_size, cell_size, 18]
    """
    out_tensor = np.ones_like(cscs, dtype=np.float32)  # initialize out_tensor
    out_tensor[cscs <= threshold] = 0
    out_tensor = np.tile(out_tensor, [1, 1, 1, predicts.shape[3]])  # expand out_tensor to the shape of predicts
    out_tensor = np.multiply(out_tensor, predicts) # multiply element-wise, when outtensor number is 0 means it will be pruned
    return out_tensor

def nms(input_tensor, cscs):
    """
    get the maximum confidence score tensor from confidence_thresh
    Args:
        cscs: class-specific confidence score  [batch_size, cell_size, cell_size, 1]
        input_tensor: out_tensor from confidence_thresh  [batch_size, cell_size, cell_size, 18]
    Return:
        output: a numpy feature tensor  [batch_size, cell_size, cell_size, 18]
    """
    cols, rows= [], []
    res = np.zeros_like(cscs, dtype=np.float32)
    for i in range(1, input_tensor.shape[1]-1, 1):
        for j in range(1, input_tensor.shape[2]-1, 1):
            temp = cscs[:, i-1:i+2, j-1:j+2, :]
            temp_max = np.argmax(temp)
            _, k, l, __ = np.where(temp==temp_max)
            res[:, k+i-1, l+j-1, :] = 1
    res = np.tile(res, [1, 1, 1, 18])
    out_tensor = np.multiply(res, input_tensor)
    return out_tensor

def compute_average(orig_tensor, cscs, out_tensor):

    return out_tensor

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
    if not os.path.exists( path ):
        os.makedirs( path )

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