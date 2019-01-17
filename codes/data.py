import os
import utils
import random
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from utils import log
from datetime import datetime
from scipy.ndimage import rotate
from keras.preprocessing.image import Iterator

class Dataset(object):
    def __init__(self, FLAGS, LOGGER):
        self.input_size = (33,33)
        self.FLAGS = FLAGS
        self.batch_size = FLAGS.batch_size
        self.image_dir = os.path.join('..','DataTrain','Images','input')
        self.label_dir = os.path.join('..','DataTrain','Images','label')
        self.checkdata(self.image_dir)
        self.test_dir = os.path.join('..','DataTest')
        self.train_names = utils.all_files_under(self.image_dir, extension='bmp',append_path=False)
        self.test_names = utils.all_files_under(self.test_dir, extension='bmp',append_path=False)
        #%% implement batch fetcher
        self.train_batch_fetcher = TrainBatchFetcher(self.train_names, self.batch_size)

   
    def train_next_batch(self):
        batch_names = self.train_batch_fetcher.next()
        images_y, labels_y, images_cbcr = self.get_imgs(batch_names, self.input_size)
        
        return np.expand_dims(images_y,axis=-1), np.expand_dims(labels_y,axis=-1)


    def get_imgs(self, file_names, mode='train'):
        n_files = len(file_names)

        image_files = []    # input image
        label_files = []    # train label

        for i in range(n_files):
            image_files.append(os.path.join(self.image_dir,"{}".format(file_names[i])))
            label_files.append(os.path.join(self.label_dir,"{}".format(file_names[i])))
        
        images_ycbcr = self.imagefiles2arrs(image_files)/255.0
        labels_ycbcr = self.imagefiles2arrs(label_files)/255.0
        images_y = images_ycbcr[...,0]
        labels_y = labels_ycbcr[...,0]
        images_cbcr = images_ycbcr[...,1:3]
        
        return images_y, labels_y, images_cbcr

    #%% get data arrays as YCbCr mode
    def imagefiles2arrs(self, filenames):
        img_shape = self.image_shape(filenames[0])
        if len(img_shape)==3:
            images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
        elif len(img_shape)==2:
            images_arr = np.zeros((len(filenames), img_shape[0], img_shape[1]), dtype=np.float32)

        for file_index in range(len(filenames)):
            img = Image.open(filenames[file_index])
            img_ycbcr = img.convert("YCbCr")
            images_arr[file_index,...] = np.array(img_ycbcr, dtype=np.float32)

        return images_arr

    def image_shape(self, filename):
        img = Image.open(filename)
        img_arr = np.asarray(img)
        img_shape = img_arr.shape
        return img_shape

    def checkdata(self, dir):
        if not os.path.exists(dir):
            utils.log('Creating train data')
            os.system("python preprocess.py")


class TrainBatchFetcher(Iterator):
    def __init__(self, train_names, batch_size):
        self.train_names = train_names
        self.n_train_names = len(self.train_names)
        self.batch_size = batch_size

    def next(self):
        indices=list(np.random.choice(self.n_train_names, self.batch_size))
        self.train_batch_names = []
        for i in indices:
            self.train_batch_names.append(self.train_names[i])

        return self.train_batch_names