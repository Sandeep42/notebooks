# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 07:20:51 2016

@author: Sandeep
"""

from __future__ import division
import logging
import os
import fnmatch
from tqdm import tqdm
import numpy as np
import cv2


logger = logging.getLogger(__name__)

def load_img(path):
    im = cv2.imread(path)
    im = im/255.0
    #im = np.expand_dims(im, axis=0)
    
#    im = np.transpose(im,(2,0,1))
#    mean = np.mean(im,axis=0)
#    im = im -mean
#    std = np.std(im,axis=0)
#    im = im - std
    return im
    
def load_npy(path):
    arr = np.load(path)
#arr = np.expand_dims(arr, axis=0)
    return arr

def get_training_images(folder, pattern="*.jpg"):
    for root, _, fns in os.walk(folder):
        logger.info("Loading training images in %s" % root)
        matches = fnmatch.filter(fns,pattern)
        for f in tqdm(matches):
            lbl = os.path.split(root)[1][-1]
            path = os.path.join(root, f)
            yield lbl, path
    
            
def load_train_resized(imgs_folder):  
    X, Y= [], []
    path = os.path.join(imgs_folder, 'train')
    for lbl, fn in get_training_images(path):
        X.append(load_img(fn))
        Y.append(lbl)
    
    X = np.array(X)
    Y = np.array(Y).astype('int32')
    
    return X,Y
    
def load_test_resized(imgs_folder):  
    X, Y= [], []
    path = os.path.join(imgs_folder, 'test')
    for lbl, fn in get_training_images(path):
        X.append(load_img(fn))
        Y.append(lbl)
    
    X = np.array(X)
    Y = np.array(Y).astype('int32')
    
    return X,Y
    

    