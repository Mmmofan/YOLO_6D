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
    predict_x = tf.stack([x1[:,:,:,0], x1[:,:,:,2], x1[:,:,:,4], x1[:,:,:,6], 
                            x1[:,:,:,8], x1[:,:,:,10], x1[:,:,:,12],  x1[:,:,:,14], x1[:,:,:,16]], 3)
    predict_y = tf.stack([x1[:,:,:,1], x1[:,:,:,3], x1[:,:,:,5], x1[:,:,:,7], 
                            x1[:,:,:,9], x1[:,:,:,11], x1[:,:,:,13],  x1[:,:,:,15], x1[:,:,:,17]], 3)
    gt_x = tf.stack([x2[:,:,:,0], x2[:,:,:,2], x2[:,:,:,4], x2[:,:,:,6], 
                        x2[:,:,:,8], x2[:,:,:,10], x2[:,:,:,12], x2[:,:,:,14], x2[:,:,:,16]], 3)
    gt_y = tf.stack([x2[:,:,:,1], x2[:,:,:,3], x2[:,:,:,5], x2[:,:,:,7], 
                        x2[:,:,:,9], x2[:,:,:,11], x2[:,:,:,13], x2[:,:,:,15], x2[:,:,:,17]], 3)
    dist = tf.sqrt(tf.add(tf.square(predict_x - gt_x), tf.square(predict_y - gt_y)))
    ### dist: 4-D tensor [batch_size, cell_size, cell_size, 9]
    return dist

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
