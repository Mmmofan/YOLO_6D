import tensorflow as tf
import numpy as np
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
    confidence = tf.reduce_mean(confidence, 3, keep_dims=True)
    return confidence

def dist(x1, x2, name='Distance'):
    """
    Args:
        x1: 4-D tensor, [batch_size, cell_size, cell_size, 18]
        x2: 4-D tensor, [batch_size, cell_size, cell_size, 18]
    Return:
    """
    predict_x = tf.stack(x1[:,:,:,0], x1[:,:,:,2], x1[:,:,:,4], x1[:,:,:,6], x1[:,:,:,8], x1[:,:,:,10], x1[:,:,:,12],  x1[:,:,:,14], x1[:,:,:,16])
    predict_y = tf.stack(x1[:,:,:,1], x1[:,:,:,3], x1[:,:,:,5], x1[:,:,:,7], x1[:,:,:,9], x1[:,:,:,11], x1[:,:,:,13],  x1[:,:,:,15], x1[:,:,:,17])
    gt_x = tf.stack(x2[:,:,:,0], x2[:,:,:,2], x2[:,:,:,4], x2[:,:,:,6], x2[:,:,:,8], x2[:,:,:,10], x2[:,:,:,12], x2[:,:,:,14], x2[:,:,:,16])
    gt_y = tf.stack(x2[:,:,:,1], x2[:,:,:,3], x2[:,:,:,5], x2[:,:,:,7], x2[:,:,:,9], x2[:,:,:,11], x2[:,:,:,13], x2[:,:,:,15], x2[:,:,:,17])
    dist = tf.sqrt(tf.add(tf.square(predict_x - gt_x), tf.square(predict_y - gt_y)))
    ### dist: 4-D tensor [batch_size, cell_size, cell_size, 9]
    return dist

def calcu_iou(boxes1, boxes2, scope='iou'):
    """
    calculate 2 boxes' iou, used in YOLO, but NOT USED IN YOLO-6D
    Args: 
        boxes1: 4-D tensor [batch_size, cell_size, cell_size, 18] ===> [x_center, y_center, x1, y1, ..., x8, y8]
        boxes2: 4-D tensor [batch_size, cell_size, cell_size, 18] ===> [x_center, y_center, x1, y1, ..., x8, y8]
    Return: 
        iou: 4-D tensor [batch_size, cell_size, cell_size, 1]
    """
    with tf.variable_scope(scope):
        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                            boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                            boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                            boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                            boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                            boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                            boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])
        
