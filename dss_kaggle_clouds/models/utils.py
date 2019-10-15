# utils.py

import numpy as np

def dice_coef(y_true, y_pred):
    """
    Parameters:
    y_true (ndarray): True segmenation of image as a 2D array
    y_pred (ndarray): Predicted segmentation of image as a 2D array
    
    Returns: 
    Dice coefficient between predicted segmentation and ground truth
    """
    y_true = np.ndarray.flatten(y_true)
    y_pred = np.ndarray.flatten(y_pred)
    return 2 * np.logical_and(y_true, y_pred) / (len(y_true), len(y_pred))


# a more efficient implementation might be useful
def dice_avg(masks_true, masks_pred):
    """
    Parameters:
    masks_true (List): List of true segmentations of images as 2D arrays
    masks_pred (List): List of predicted segmentations of images as 2D arrats

    Returns:
    Average Dice coefficient over the dataset
    """ 

    coeff = 0
    for y_true, y_pred in zip(masks_true, masks_true):
        y_true = np.ndarray.flatten(y_true)
        y_pred = np.ndarray.flatten(y_pred)
        coeff += 2 * np.logical_and(y_true, y_pred) / (len(y_true), len(y_pred))

    return coeff