from tensorflow.keras.backend import batch_flatten, epsilon
import numpy as np
import tensorflow as tf

'''Implementation of the proposed Sharpening Loss function.
   The Sharpening Loss results in sharper foreground objects and less blurry predictions.
   Moreover, as shown in the paper, the Sharpening Loss outperforms the Cross-entropy loss
   by a significant margin in saliency detetction task'''


def Sharpenning_Loss(y_true, y_pred):
    
    beta2 = tf.constant(0.3, dtype='float32')
    eps   = tf.constant(1e-22, dtype='float32')
    
    y_true = tf.cast(y_true, tf.float32)  

    y_true = batch_flatten(y_true)
    y_pred = batch_flatten(y_pred)
    
    P = tf.reduce_sum(y_pred * y_true, axis = -1) / (tf.reduce_sum(y_pred, axis = -1) + eps)
    R = tf.reduce_sum(y_pred * y_true, axis = -1) / (tf.reduce_sum(y_true, axis = -1) + eps)        
    
    L_F = 1-(((1+beta2)*tf.reduce_mean(P)*tf.reduce_mean(R)) / ((beta2*tf.reduce_mean(P))+tf.reduce_mean(R)+eps))
    
    L_MAE = tf.reduce_mean(tf.abs(y_pred-y_true))
    
    return L_F + 1.75 * L_MAE


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def cross_entropy_balanced(y_true, y_pred):
    """ tensorflow 2.0
    Implements Equation [2] in https://arxiv.org/pdf/1504.06375.pdf
    Compute edge pixels for each training sample and set as pos_weights to tf.nn.weighted_cross_entropy_with_logits
    """
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)
    alpha = 1.1 * count_pos / (count_neg + count_pos)
    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divided by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)