# utils.py

import tensorflow as tf

def dice_coeff(y_true, y_pred):
    """
    Parameters:

    y_true (ndarray): True segmenation of image as a 2D array

    y_pred (ndarray): Predicted segmentation of image as a 2D array
    
    Returns: 

    Dice coefficient between predicted segmentation and ground truth
    """
    y_true = tf.reshape(y_true, [-1]) # a shape of -1 flattens to 1D
    y_pred = tf.reshape(y_pred, [-1])
    return 2 * tf.logical_and(y_true, y_pred) / (len(y_true), len(y_pred))

