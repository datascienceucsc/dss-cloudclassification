# data_loader.py
#
# Anders Poirel
# 26-10-2019
#
# main script for training the model

import tensorflow as tf
import pandas as pd
import numpy as np
from dss_kaggle_clouds.models.unet import U_net
from dss_kaggle_clouds.models.metrics import dice_coeff

if __name__ == 'main':
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        rotation_range=90,
                        horizontal_flip=True,
                        vertical_flip=True)

    mask_datagen = tf.keras.preprocessing.image.ImageDataGeneratorr(
                        rescale=1./255,
                        shear_range=0.2,
                        zoom_range=0.2,
                        rotation_range=90,
                        horizontal_flip=True,
                        vertical_flip=True)

    train_image_generator = train_datagen.flow_from_directory(
                                '../data/raw/train_images',
                                target_size = (1400, 2100),
                                class_mode = None,
                                batch_size = 4)

    train_mask_generator = train_datagen.flow_from_directory(
                                '../data/processed',
                                target_size = (1400, 2100),
                                classes = ['mask_fish', 'mask_flower',
                                           'mask_gravel', 'mask_sugar'],
                                class_mode = None,
                                color_mode = 'grayscale',
                                batch_size = 4)

    model = U_net('SGD', 'relu', dice_coeff)
    history = model.fit(train_image_generator, train_mask_generator,
              epochs = 10, validation_split = 0.2, workers = -1)

    # saves outputs of the model for diagnostic and retraining

    hist_df = pd.DataFrame(history.history) 
    hist_df.to_csv('../../models/training_history.csv')

    tf.saved_model.save('../../models/1/')

    # uncomment following code to generate predictions on a local machine
    #
    # test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    # test_image_generator =  test_datagen.flow_from_directory(
    #                            '../data/raw/test_images',
    #                            target_size = (1400, 2100),
    #                            class_mode = None)

    # predictions = model.predict(test_image_generator)

    # saves the predictions for building the kaggle submission locallly
    

