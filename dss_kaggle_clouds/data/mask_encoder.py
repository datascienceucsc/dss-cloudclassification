# mask_encoder.py
#
# Anders Poirel
# 
# fucntions to a .csv file with the (compressed) encoded masks for each 
# image-label pair
 
import pandas as pd
import numpy as np
import os
import PIL

 
def encode_mask(img):
    """
    Parameters

    - img (2d array): flattened array representing the mask for a given image
    
    Returns: string representation of the mask
    """

    str_rep = []
    in_mask = False
    for px in np.nditer(img, flags = ['f_index']):
        if (not in_mask) and px == 1:
            in_mask = True
            starting = px.index
            count = 1
        elif in_mask and px == 1:
            count += 1
        else: # we just reached the end of a sequence of masked pixels
            str_rep.append("%d %d " % (starting, count))

    # this should return an empty string if no masks are found
    return ''.join(str_rep)


if __name__ == 'main':
    submission = pd.read_csv('../../data/raw/sample_submission.csv')
    masks  = [np.array(PIL.Image.open('../../models/%s' % filename))
             for filename in os.listdir('../../models')]
    submission['EncodedPixels'] = map(encode_mask, masks)
    
    submission.to_csv(index = False)