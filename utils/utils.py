# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import cv2
import config as cfg

def confidence_func(x, name='Confidence func'):
    """
    input: Two 4-D tensor: [Batch_size, feature_size, feature_size, 18]
    compute confidence score then concat to original
    output: A 4-D tensor: [Batch_Size, feature_size, feature_size, 18]
    """
    alpha = tf.constant(cfg.ALPHA, dtype=tf.float32)
    dth_in_cell_size = tf.constant(cfg.Dth / cfg.CELL_SIZE, dtype=tf.float32)
    param1 = tf.ones_like(x, dtype=tf.float32)
    confidence = (tf.exp(alpha * (param1 - x / dth_in_cell_size))) / (tf.exp(alpha) - param1)
    confidence = tf.reduce_mean(confidence, 3, keepdims=True)
    return confidence

def dist(x1, x2, name='Distance'):
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
    dist = tf.sqrt(tf.add(tf.square(predict_x - gt_x), tf.square(predict_y - gt_y)))
    ### dist: 4-D tensor [batch_size, cell_size, cell_size, 9]
    return dist

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
                                (1, 2, 0))
    off_set = np.array(off_set).astype(np.float32)
    off_set = np.reshape(off_set, [1, cfg.CELL_SIZE, cfg.CELL_SIZE, 18 * cfg.BOXES_PER_CELL])
    off_set = np.tile(off_set, [cfg.BATCH_SIZE, 1, 1, 1])
    ## off_set shape : [Batch_Size, cell_size, cell_size, 18 * boxes_per_cell]
    off_set_centroids = off_set[:, :, :, :2]
    off_set_corners = off_set[:, :, :, 2:]
    predict_boxes_tran = np.concatenate([tf.add(tf.nn.sigmoid(coordinates[:, :, :, :2]), off_set_centroids),
                                        tf.add(coordinates[:, :, :, 2:], off_set_corners)], 3)
    predict_boxes = np.multiply(predict_boxes_tran, tf.constant(float(cfg.CELL_SIZE)))
    
    # Cut the box, assert the box bounding less than 416
    boxes_max_min = np.array([0, 0, 416, 416], dtype=np.float32)
    predict_boxes = bboxes_cut(boxes_max_min, predict_boxes)



def bboxes_cut(bbox_min_max, bboxes):
    #unfinish
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    bbox_min_max= np.transpose(bbox_min_max)
    #cut the boxes
    bboxes[0] = np.maximum(bboxes[0], bbox_min_max[0]) #xmin
    bboxes[1] = np.maximum(bboxes[1], bbox_min_max[1]) #ymin
    bboxes[2] = np.minimum(bboxes[2], bbox_min_max[2]) #xmax
    bboxes[3] = np.minimum(bboxes[3], bbox_min_max[3]) #ymax
    bboxes = np.transpose(bboxes)
    return bboxes

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
    out_tensor = np.zeros_like(predicts)
    for i in range(cfg.CELL_SIZE):
        for j in range(cfg.CELL_SIZE):
            for k in range(cfg.BATCH_SIZE):
                if cscs[k][i][j][0] > threshold:
                    out_tensor[k][i][j] = predicts[k][i][j]
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
    out_tensor = np.zeros_like(input_tensor)
    cols, rows = [], []
    for i in range(1, cfg.CELL_SIZE-1, 1):
        for j in range(1, cfg.CELL_SIZE-1, 1):
            col = np.argmax(cscs[:, i-1:i+2, j-1:j+2, 0], axis=0)
            row = np.argmax(cscs[:, i-1:i+2, j-1:j+2, 0], axis=1)
    
    return out_tensor

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
