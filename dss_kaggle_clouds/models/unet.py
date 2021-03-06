# unet.py
#
# Anders Poirel
# 25-10-2019
#
# Implementation of the U-net architecture for image segmentation for the
# Kaggle cloud classifaction competition. Note that the code is somewhat
# specific to the image sizes in this compeitition and would require a 
# fair bit of tweaking to adapt to other problems.


import tensorflow as tf
import pdb
from tensorflow.keras.layers import (Dense, Conv2D, Conv2DTranspose, 
    MaxPool2D, concatenate, ZeroPadding2D)


def U_net(optimizer, activation, metrics):
    """
    Parameters:
    - optimizer (String): Keras optimizer to use
    - activation (String): Keras layer activation function to use in hidden 
    layers
    - metrics to use for model evaluation

    Returns: (Model)
    a compiled U-net designed for binary segmentation.
    """
    
    inputs = tf.keras.Input(shape = (2100, 1400, 3))
    # downsampling layers

    conv_1 = Conv2D(64, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(inputs)
    conv_2 = Conv2D(64, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    pool_1 = MaxPool2D((2,2))(conv_2)

    conv_3 = Conv2D(128, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(pool_1)
    conv_4 = Conv2D(128, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    pool_2 = MaxPool2D((2,2))(conv_4)

    conv_5 = Conv2D(256, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(pool_2)
    conv_6 = Conv2D(256, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(conv_5)
    pool_3 = MaxPool2D((2,2))(conv_6)

    conv_7 = Conv2D(512, (3,3), activation = activation,
                    kernel_initializer = 'he_normal', padding = 'same')(pool_3) 
    conv_8 = Conv2D(512, (3,3), activation = activation,
                    kernel_initializer = 'he_normal', padding = 'same')(conv_7)
    pool_4 = MaxPool2D((2,2))(conv_8)
    
    conv_9 = Conv2D(1024, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(pool_4)
    conv_10 = Conv2D(1024, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(conv_9)

    # upsampling layers
    
    upconv_1 = Conv2DTranspose(512, (2,2), strides = (2, 2), padding = 'same')(conv_10)
    upconv_1 = ZeroPadding2D(padding = ((0, 0), (0, 1)))(upconv_1)  # we need to padd with zeroes on the right 
    merge_1 = concatenate([conv_8, upconv_1]) 
    conv_11 = Conv2D(512, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(merge_1)
    conv_12 = Conv2D(512, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(conv_11)


    upconv_2 = Conv2DTranspose(256, (2, 2), strides = (2, 2), padding='same')(conv_12)
    upconv_2 = ZeroPadding2D(padding = ((0,1),(0,0)))(upconv_2) # we need to padd with zeroes on top
    merge_2 = concatenate([conv_6, upconv_2])
    conv_13 = Conv2D(256, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(merge_2)
    conv_14 = Conv2D(256, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(conv_13)

    upconv_3 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding='same')(conv_14)
    merge_3 = concatenate([conv_4, upconv_3])
    conv_15 = Conv2D(128, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(merge_3)
    conv_16 = Conv2D(128, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(conv_15)

    upconv_4 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding='same')(conv_16)
    merge_4 = concatenate([conv_2, upconv_4])
    conv_17 = Conv2D(64, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(merge_4)
    conv_18 = Conv2D(64, (3,3), activation = activation,
                     kernel_initializer = 'he_normal', padding = 'same')(conv_17) 

    output = Dense(4, activation = 'softmax')(conv_18)

    model = tf.keras.Model(inputs, output)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy',
                  metrics = metrics)
    return model
s