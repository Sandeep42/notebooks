# In[1]:

import boilerplate
from boilerplate import keras_resized_code


# In[4]:

from skimage.io import imread, imshow
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import randint
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from keras.utils import np_utils

import cv2
import h5py
import os
import h5py


# In[3]:

X_train, y_train = keras_resized_code.load_train_resized('resized/')


# In[5]:

X_train.shape


# In[6]:

X_test, y_test = keras_resized_code.load_test_resized('resized/')


# In[7]:

X_test.shape


# In[8]:

imshow(X_train[0])


# In[9]:

imshow(X_test[0])


# In[10]:

X_test = np.transpose(X_test, (0,3,1,2))


# In[11]:

X_train = np.transpose(X_train, (0,3,1,2))


# In[12]:

X_test.shape


# In[13]:

X_train.shape


# In[14]:



# In[20]:

y_train = np_utils.to_categorical(y_train)


# In[14]:

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))

# Add another conv layer with ReLU + GAP
model.add(Convolution2D(1024, 3, 3, activation='relu', border_mode="same"))
model.add(AveragePooling2D((14, 14)))
model.add(Flatten())
# Add the W layer
model.add(Dense(10, activation='softmax'))
with h5py.File('pretrained_models/vgg16_weights.h5') as hw:
    for k in range(hw.attrs['nb_layers']):
        g = hw['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
        if model.layers[k].name == "convolution2d_13":
            break
    print('Model loaded.')

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)   
model.compile(sgd, loss='categorical_crossentropy',metrics=['accuracy'])


# In[15]:

index = randint(15213, size= 400)


# In[16]:

X_train_sample = X_train[index,:,:,:]


# In[17]:

X_train_sample.shape


# In[18]:

y_train.shape


# In[21]:

y_train_sample = y_train[index,:]


# In[22]:

y_train_sample.shape


# In[ ]:

model.fit(X_train_sample, y_train_sample)


# In[23]:




# In[24]:




# In[ ]:



