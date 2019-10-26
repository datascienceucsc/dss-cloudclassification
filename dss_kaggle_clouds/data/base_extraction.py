import pandas as pd    
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.io as io
import skimage.transform as tran

def rle2mask(row, height = 1400, width = 2100):
    '''
    convert RLE(run length encoding) string to numpy array

    Parameters: 
    rle_string (str): string of rle encoded mask
    height (int): height of the mask
    width (int): width of the mask 

    Returns: 
    numpy.array: numpy array of the mask
    '''
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
    mask_file = row['Label'] + '_' + row['Image'] + '.jpg'
    
    if label == 'Fish':
        pil_image.save('../../data/processed/mask_fish/' + mask_file)
    elif label == 'Flower':
        pil_image.save('../../data/processed/mask_flower/' + mask_file)
    elif label == 'Gravel':
        pil_image.save('../../data/processed/mask_gravel' + mask_file)
    elif label == 'Sugar':
        pil_image.save('../../data/processed/mask_sugar' + mask_file)

train = pd.read_csv('../../data/raw/train.csv')
train = train.fillna(-1)
train[['Image', 'Label']] = train['Image_Label'].str.split('_', expand=True)
train['Image'] = train['Image'].str.split('.').str[0]
train = train.drop('Image_Label', axis=1)
train.apply(rle2mask, axis = 1)