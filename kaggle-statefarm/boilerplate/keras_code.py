# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 09:11:24 2016

@author: Sandeep
"""
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(rotation_range=50,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             shear_range=0.2,
                             zoom_range=0.4,
                             horizontal_flip=True,
                             fill_mode='nearest')

img = load_img('input/images/train/c0/img_34')
x = img_to_array(img)
x = x.reshape((1,)+ x.shape)

i=0

for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely              
