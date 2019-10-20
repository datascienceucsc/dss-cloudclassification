import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Flatten

    
# Con2DTranpose does deconvolution/upsampling
# I try to implement the Unet32 architecture

# This Unet is used for binary classification only

def U_net(optimizer, metrics, activation):
    """
    Returns a U-net designed for the problem.
    """

    inputs = tf.keras.Input(shape = (2100, 1400, 1))

    # downsampling layers

    conv_1 = Conv2D(64, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(inputs)
    conv_2 = Conv2D(64, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(conv_1)
    pool_1 = MaxPooling2D((2,2))(conv_2)

    conv_3 = Conv2D(128, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(pool_1)
    conv_4 = Conv2D(128, (3,3), activation = activation,
               kernel_initializer = 'he_normal', padding = 'same')(conv_3)
    pool_2 = MaxPooling2D((2,2))(conv_4)
    conv_5 = Conv2D(256, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(pool_2)
    conv_6 = Conv2D(256, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(conv_5)
    pool_3 = MaxPooling2D((2,2))(conv_4)
    conv_7 = Conv2D(512, (3,3), activation = activation,
                    kernel_initializer = 'he_normal', padding = 'same')(pool_3) 
    conv_8 = Conv2D(512, (3,3), activation = activation,
                    kernel_initializer = 'he_normal', padding = 'same')(conv_7) 
    pool_4 = MaxPooling2D((2,2))(conv_8)
    conv_9 = Conv2D(1024, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(pool_4)
    conv_10 = Conv2D(1024, (3,3), activation = activation,
                   kernel_initializer = 'he_normal', padding = 'same')(conv_9)

    print(conv_10.shape)
    # upsampling layers
    
    upconv_1 = Conv2DTranspose(512, (2,2), strides = (2, 2), padding = 'same')(conv_10)
    comb_1 = concatenate([conv_8, upconv_1]) 
    conv_11 = Conv2D(512, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same') (comb_1)
    conv_12 = Conv2D(512, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(conv_11)
    upconv_2 = Conv2DTranspose(256, (2, 2), strides = (2, 2), padding='same')(conv_12)
    comb_2 = concatenate([conv_6, upconv_2])
    conv_13 = Conv2D(256, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(comb_2)
    conv_14 = Conv2D(256, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(conv_13)
    upconv_3 = Conv2DTranspose(128, (2, 2), strides = (2, 2), padding='same')(conv_14)
    comb_3 = concatenate([conv_4, upconv_3])
    conv_15 = Conv2D(128, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(comb_3)
    conv_16 = Conv2D(128, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(conv_15)
    upconv_4 = Conv2DTranspose(64, (2, 2), strides = (2, 2), padding='same')(conv_16)
    comb_3 = concatenate([conv_2, upconv_4])
    conv_17 = Conv2D(64, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(comb_3)
    conv_18 = Conv2D(64, (3,3), activation = activation,
                     kernel_initializer = activation, padding = 'same')(conv_17) 
    output = Conv2D(1, (1,1), activation = 'sigmoid')(conv_18)

    model = tf.keras.Model(inputs, output)
    model.compile(optimizer = optimizer, loss = 'BinaryCrossentropy', metrics = metrics)
    return model


model = U_net(optimizer = 'SGD', metrics = ['accuracy'], activation = 'sigmoid')
