from __future__ import division
import logging
import os
import fnmatch
from tqdm import tqdm


#from skimage.io import imread, imsave
#from skimage.transform import resize
#from skimage import img_as_ubyte, img_as_float
import h5py
import cv2
import numpy as np
import pandas as pd
from sklearn.cross_validation import LabelKFold
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)

MEAN_VALUE = np.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))

SIZE_X = 224
SIZE_Y = 224



def get_training_images(folder, drivers, pattern="*.jpg"):
    for root, _, fns in os.walk(folder):
        logger.info("Loading training images in %s" % root)
        matches = fnmatch.filter(fns,pattern)
        for f in tqdm(matches):
            lbl = os.path.split(root)[1][-1]
            driver = drivers[f]
            path = os.path.join(root, f)
            yield lbl, driver, path
    
def get_test_images(folder, pattern="*.jpg"):
    for root, _, fns in os.walk(folder):
        logger.info("Loading test images in %s" % root)
        matches = fnmatch.filter(fns, pattern)
        for f in tqdm(matches):
            path = os.path.join(root,f)
            yield path 

def load_drivers(path):
    df = pd.read_csv(path)
    drivers = {fn: d for fn, d in zip(df['img'],df['subject'])}
    return drivers

def load_train(imgs_folder, drivers_path):
    drivers = load_drivers(drivers_path)
    
    X, Y, D = [], [], []
    path = os.path.join(imgs_folder, 'train')
    for lbl, driver, fn in get_training_images(path, drivers):
        X.append(fn)
        Y.append(lbl)
        D.append(driver)
    #logger.info('Loaded %d images' % len(Y))
    
    X = np.array(X)
    Y = np.array(Y).astype('int32')
    D = np.array(D)
    
    logger.info('Splitting local validaton set')
    X, Y, D = shuffle(X, Y, D)
    cv = LabelKFold(D,3)
    index_s, index_t = next(iter(cv))
    
    Xs, Xt = X[index_s], X[index_t]
    Ys, Yt = Y[index_s], Y[index_t]
    
    return Xs, Ys, Xt, Yt
    


def load_train_resized(imgs_folder, drivers_path):
    drivers = load_drivers(drivers_path)
    
    X, Y, D = [], [], []
    path = os.path.join(imgs_folder, 'train')
    for lbl, driver, fn in get_training_images(path, drivers):
        X.append(fn)
        Y.append(lbl)
        D.append(driver)
    #logger.info('Loaded %d images' % len(Y))
    
    X = np.array(X)
    Y = np.array(Y).astype('int32')
    D = np.array(D)
    
    return X,Y
    
def load_test_resized(imgs_folder, drivers_path):
    drivers = load_drivers(drivers_path)
    
    X, Y, D = [], [], []
    path = os.path.join(imgs_folder, 'test')
    for lbl, driver, fn in get_training_images(path, drivers):
        X.append(fn)
        Y.append(lbl)
        D.append(driver)
    #logger.info('Loaded %d images' % len(Y))
    
    X = np.array(X)
    Y = np.array(Y).astype('int32')
    D = np.array(D)
    
    return X,Y
    
def im_yield(path):
    im = cv2.resize(cv2.imread(path), (224, 224)).astype(np.float32)
    #im[:,:,0] -= 103.939
    #im[:,:,1] -= 116.779
    #im[:,:,2] -= 123.68
    #im = im.transpose((2,0,1))
    return im

    
#def image_yield_sk(path):
#    im = imread(path)
#    im = resize(im, (224, 224))    
#    im = im.astype(np.float32)
#    im[:,:,0] -= 103.939
#    im[:,:,1] -= 116.779
#    im[:,:,2] -= 123.68
#    im = img_as_ubyte(im)
#    #im = img_as_float(im)
#    #im = im.transpose((2,0,1))
#    return im

def store_imgs():
    Xs, Ys, Xt, Yt = load_train('input/imgs/','input/driver_imgs_list.csv')    
    count_train = 0
    for path, lbl in tqdm(zip(Xs, Ys)):        
        im_array = image_yield_sk(path)
        if not os.path.exists('resized'):
            os.makedirs('resized')
        if not os.path.exists('resized/train'):
            os.makedirs('resized/train')
        check_label = os.path.join('resized','train','c'+ str(lbl))
        if not os.path.exists(check_label):
            os.makedirs(check_label)
        suffix = '.npy'
        filename = os.path.join('resized','train','c'+ str(lbl), str(count_train) + suffix)
        np.save(filename,im_array)
        count_train += 1
def store_imgs_test():
    Xs, Ys, Xt, Yt = load_train('input/imgs/','input/driver_imgs_list.csv')   
    count_test = 0
    for path, lbl in tqdm(zip(Xt, Yt)):        
        im_array = image_yield_sk(path)
        if not os.path.exists('resized'):
            os.makedirs('resized')
        if not os.path.exists('resized/test'):
            os.makedirs('resized/test')
        check_label = os.path.join('resized','test','c'+ str(lbl))
        if not os.path.exists(check_label):
            os.makedirs(check_label)
        suffix = '.npy'
        filename = os.path.join('resized','test','c'+ str(lbl), str(count_test) + suffix)
        np.save(filename,im_array)
        count_test += 1
        
    im_array = np.array(im_array)
    lbl = np.array(lbl)
    # dump to h5py
    with h5py.File('imgs.h5', 'w') as hf:
        hf.create_dataset('X_train', data=im_array)
        hf.create_dataset('y_train', data=lbl)


    
def write_images():
    Xs, Ys, Xt, Yt = load_train('input/imgs/','input/driver_imgs_list.csv')
    count_train = 0
    for path, lbl in tqdm(zip(Xs, Ys)):        
        #print lbl
        im_array = im_yield(path)
        if not os.path.exists('resized'):
            os.makedirs('resized')
        if not os.path.exists('resized/train'):
            os.makedirs('resized/train')
        check_label = os.path.join('resized','train','c'+ str(lbl))
        if not os.path.exists(check_label):
            os.makedirs(check_label)
        suffix = '.jpg'
        filename = os.path.join('resized','train','c'+ str(lbl), str(count_train) + suffix)
        #filename = os.path.join('resized','train',str(count),'*.jpg')
        cv2.imwrite(filename,im_array)
        count_train += 1
    count_test = 0
    for path, lbl in tqdm(zip(Xt, Yt)):        
        #print lbl
        im_array = im_yield(path)
        if not os.path.exists('resized'):
            os.makedirs('resized')
        if not os.path.exists('resized/test'):
            os.makedirs('resized/test')
        check_label = os.path.join('resized','test','c'+ str(lbl))
        if not os.path.exists(check_label):
            os.makedirs(check_label)
        suffix = '.jpg'
        filename = os.path.join('resized','test','c'+str(lbl), str(count_test) + suffix)
        #filename = os.path.join('resized','train',str(count),'*.jpg')
        cv2.imwrite(filename,im_array)
        count_test += 1    
        
        
    
    
def load_test(imgs_folder):
    X, fnames  = [], []
    path = os.path.join(imgs_folder, 'test')
    for fn in get_test_images(path):
        X.append(fn)
        fnames.append(os.path.split(fn)[-1])
    logger.info("Loaded %d images" % len(X))

    X = np.array(X)
    fnames = np.array(fnames)
    logger.info("Xt shape: (%d,)" % X.shape)
    logger.info("fnames shape: (%d,)" % fnames.shape)

    return X, fnames