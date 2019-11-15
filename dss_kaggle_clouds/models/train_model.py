# train_model.py
#
# Anders Poirel
# 26-10-2019
#
# main script for training the model

#%%
import tensorflow as tf
import pandas as pd
from metrics import dice_coeff, dice_loss
from unet import U_net

#%%
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        rescale = 1./255,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        rotation_range = 90,
                        horizontal_flip = True,
                        vertical_flip = True,
                        validation_split = 0.2)

mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                        rescale = 1./255,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        rotation_range = 90,
                        horizontal_flip = True,
                        vertical_flip = True,
                        validation_split = 0.2)

#%%
train_image_generator = train_datagen.flow_from_directory(
                                '../../data/raw/train_images',
                                target_size = (1400, 2100),
                                class_mode = None,
                                batch_size = 16)

train_mask_generator = train_datagen.flow_from_directory(
                                '../../data/processed',
                                target_size = (1400, 2100),
                                classes = ['mask_fish', 'mask_flower',
                                           'mask_gravel', 'mask_sugar'],
                                class_mode = None,
                                color_mode = 'grayscale',
                                batch_size = 16)

#%%
train_generator = (pair for pair in
     zip(train_image_generator, train_mask_generator))

#%%
model = U_net(optimizer = 'SGD', activation = 'relu',
              loss = dice_loss, metrics = [dice_coeff])

#%%
history = model.fit_generator(train_generator, epochs = 10,
                              steps_per_epoch = sum(1 for x in train_generator),
                              workers = -1)

# saves outputs of the model for diagnostic and retraining

#%%
hist_df = pd.DataFrame(history.history) 
hist_df.to_csv('../../models/training_history.csv')

tf.saved_model.save('../../models/1/')

# uncomment following code to generate predictions on a local machine
# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
# test_image_generator =  test_datagen.flow_from_directory(
#                            '../data/raw/test_images',
#                            target_size = (1400, 2100),
#                            class_mode = None)

# predictions = model.predict(test_image_generator)

# saves the predictions for building the kaggle submission locallly
    



# %%
