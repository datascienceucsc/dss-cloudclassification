# mask_decoder.py
# 
# Sampson Mao, Anders Poirel
# 26-10-2019
#
# script for decoding the masks for each image in the training set from
#  the .csv file, writing the output as a seperate .jpg file for each 
# label-image pair

import pandas as pd    
import numpy as np
from PIL import Image

def rle2mask(row, height = 1400, width = 2100):
    """
    Parameters:

    - row (Int): row being decoded in the csv file
    - height (Int): height of image being decoded 
    - width (Int): width of image being decoded

    Generates a mask from a row in the csv file for an image in the training
    set, writing a .jpg file to the appropriate folder for the label
    """

    if row['EncodedPixels'] == -1:
        return
    rows, cols = height, width
    rle_string = row['EncodedPixels']
    
    rle_numbers = [int(num_string) for num_string 
                   in rle_string.split(' ')]
    rle_pairs = np.array(rle_numbers).reshape(-1,2)
    img = np.zeros(rows * cols, dtype = np.uint8)
    for index, length in rle_pairs:
        index -= 1
        img[index : index + length] = 255
    img = img.reshape(cols,rows)
    img = img.T
    
    pil_image = Image.fromarray(img)
    
    label = row['Label']
    mask_file = row['Image'] + '.jpg' + '_' + row['Label'] + '.jpg'
    
    if label == 'Fish':
        pil_image.save('../../data/processed/mask_fish/' + mask_file)
    elif label == 'Flower':
        pil_image.save('../../data/processed/mask_flower/' + mask_file)
    elif label == 'Gravel':
        pil_image.save('../../data/processed/mask_gravel/' + mask_file)
    elif label == 'Sugar':
        pil_image.save('../../data/processed/mask_sugar/' + mask_file)

if __name__ == 'main':

    train = pd.read_csv('../../data/raw/train.csv')
    train = train.fillna(-1)
    train[['Image', 'Label']] = train['Image_Label'].str.split('_', expand=True)
    train['Image'] = train['Image'].str.split('.').str[0]
    train = train.drop('Image_Label', axis=1)
    train.apply(rle2mask, axis = 1)