import os
import sys
import utils
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from data import Dataset
from utils import log
from logger import LOGGER
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers.core import Activation
from keras.optimizers import Adam
# from model import CNNModels
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#%% arrange arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default = 128, type=int, help="batch size")
parser.add_argument('--epochs', default = '20', type=int, help="nums of epoch")
parser.add_argument('--pretrained', default = False, type=bool, help="whether load pretrained model")
parser.add_argument('--save_frequency', default = 10, type=int, help="whether load pretrained model")
parser.add_argument('--gpu_index', default = '0', type=str, help="whether load pretrained model")
FLAGS, _ = parser.parse_known_args()
log('Settings')
utils.showFLAGS(FLAGS)

#%% set logger
logger = LOGGER(FLAGS)
log('Create Logger Successfully')

#%% set train data
dataset = Dataset(FLAGS, logger)
n_images = len(dataset.train_names)
inputsize = dataset.input_size


#%% create model
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_index
img_ch = 1
img_height, img_width = inputsize[0], inputsize[1]
inputs = Input((img_height, img_width, img_ch))
x = Conv2D(64, kernel_size=(9, 9), padding='valid')(inputs)
x = Activation('relu')(x)
x = Conv2D(32, kernel_size=(1, 1), padding='valid')(x)
x = Activation('relu')(x)
x = Conv2D(1,  kernel_size=(5, 5), padding='valid')(x)
x = Activation('relu')(x)
SRCNN = Model(inputs=inputs, outputs=x, name='SRCNN')
SRCNN.compile(Adam(lr=0.0002), loss='mse')

#%% log model
logger.callback.set_model(SRCNN) # record in tensorboard
logger.save_model(SRCNN, "SRCNN_model")
logger.load_pretrained_weights(SRCNN, 'SRCNN_Weights.h5')
logger.plot_model(SRCNN, 'SRCNN')
log('Create Networks Successfully')


###################################################################
######################## Start training ###########################
###################################################################
batch_steps = n_images//FLAGS.batch_size
for epoch in range(FLAGS.epochs):
    log("Epoch {}".format(epoch+1))

    testindex = np.random.choice(len(dataset.test_names),1)
    testname = os.path.join(dataset.test_dir, dataset.test_names[testindex[0]])
    img_ycbcr = np.squeeze(dataset.imagefiles2arrs([testname]))
    utils.predict(img_ycbcr/255.0, SRCNN, epoch)

    loss_sum = 0
    for i in range(batch_steps):
        x_batch, y_batch = dataset.train_next_batch()
        loss = SRCNN.train_on_batch(x_batch, y_batch)
        loss_sum += loss
        utils.progressbar(i, batch_steps)
    logger.write_tensorboard(['train_loss'], [loss_sum/i], epoch)

logger.save_weights(SRCNN, "SRCNN_Weights")