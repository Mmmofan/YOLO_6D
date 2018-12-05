import tensorflow as tf
import numpy as np

def conf_func(x1, x2, name='Confidence func'):
    """
    input: Two 4-D tensor: [Batch_size, feature_size, feature_size, 18]
    compute confidence score then concat to original
    output: A 4-D tensor: [Batch_Size, feature_size, feature_size, 19]
    """
    x1 = tf.subtract(x1, x2) 