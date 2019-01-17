import os
import sys
import pickle
import random
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

from PIL import Image

def predict(image, SRCNN, epoch, mode='compare'):
    scale = 3.0
    input_size = 33
    label_size = 21
    pad = int((input_size - label_size) / 2)

    w, h, c = image.shape
    w -= int(w % scale)
    h -= int(h % scale)
    image = image[0:w, 0:h, :]

    scaled = misc.imresize(image, 1.0/scale, 'bicubic')
    scaled = misc.imresize(scaled, scale/1.0, 'bicubic')/255.0
    newimg = np.zeros(scaled.shape)

    for i in range(0, h - input_size + 1, label_size):
        for j in range(0, w - input_size + 1, label_size):
            sub_img_y = np.expand_dims(scaled[j : j + input_size, i : i + input_size, 0],axis=-1)
            prediction = np.squeeze(SRCNN.predict(np.expand_dims(sub_img_y, axis=0)))
            prediction[prediction > 1.0] = 1.0
            newimg[j + pad : j + pad + label_size, i + pad : i + pad + label_size, 0] = prediction
    newimg[...,1:] = scaled[...,1:]
    newimg = newimg[pad : w - w % input_size, pad : h - h % input_size,:]

    if(mode == 'compare'):
        height = int(1.2*h)
        width = int(1.2*w)
        image = pad_img(ycbcr2rgb(image*255.0), [width,height])
        scaled = pad_img(ycbcr2rgb(scaled*255.0), [width,height])
        newimg = pad_img(ycbcr2rgb(newimg*255.0), [width,height])
        saveimg = np.hstack((image,scaled,newimg))
        # plt.subplot(131)
        # plt.imshow(image/255.0)
        # plt.subplot(132)
        # plt.imshow(scaled/255.0)   
        # plt.subplot(133)
        # plt.imshow(newimg/255.0)
        # plt.show()

        array2image(saveimg).save('../TrainResult/Round{}.png'.format(epoch+1))
        return None
    else:
        return newimg



def pad_img(img, img_size):
    img_h,img_w=img.shape[0], img.shape[1]
    target_h,target_w=img_size[0],img_size[1]

    if len(img.shape)==3:
        d=img.shape[2]
        padded=np.zeros((target_h, target_w,d))
    elif len(img.shape)==2:
        padded=np.zeros((target_h, target_w))

    padded[(target_h-img_h)//2:(target_h-img_h)//2+img_h,(target_w-img_w)//2:(target_w-img_w)//2+img_w,...]=img

    return padded

def array2image(arr):
    if(np.max(arr)<=1):
        image = Image.fromarray((arr*255).astype(np.uint8))
    else:
        image = Image.fromarray((arr).astype(np.uint8))

    return image


def showFLAGS(FLAGS):
    for i in vars(FLAGS):
        if len(i) < 8:
            print(i + '\t\t------------  ' + str(vars(FLAGS)[i]))
        else:
            print(i + '\t------------  ' + str(vars(FLAGS)[i]))
    print()

def progressbar(progress, total, length=40):
    total = total - 1
    num = int(progress/total*length)
    sys.stdout.write('#' * num + '_' * (length - num) )
    sys.stdout.write(':{:.2f}%'.format(progress/total*100)+ '\r')
    if progress == total:
        sys.stdout.write('\n\n')
    sys.stdout.flush()

def checkpath(path):
    try:
        os.makedirs(path)
        # print('creat ' + path)
    except OSError:
        pass

def log(text):
    """
    log status with time label
    """
    print()
    nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line1 = '='*10 + '  ' + nowTime + '  '+ '='*10
    length = len(line1)
    leftnum = int((length - 4 - len(text))/2)
    rightnum = length - 4 - len(text) - leftnum
    line2 = '*'*leftnum + ' '*2 + text  + ' '*2 + '*'*rightnum
    print(line1)
    print(line2)
    print('='*len(line1))


def all_files_under(path, extension=None, append_path=True, sort=True):
    if append_path:
        if extension is None:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(extension)]
    else:
        if extension is None:
            filenames = [os.path.basename(fname) for fname in os.listdir(path)]
        else:
            filenames = [os.path.basename(fname) for fname in os.listdir(path) if fname.endswith(extension)]

    if sort:
        filenames = sorted(filenames)

    return filenames







mat = np.array(
    [[ 65.481, 128.553, 24.966 ],
     [-37.797, -74.203, 112.0  ],
     [  112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)
offset = np.array([16, 128, 128])
 
def rgb2ycbcr(rgb_img):
    ycbcr_img = np.zeros(rgb_img.shape)
    for x in range(rgb_img.shape[0]):
        for y in range(rgb_img.shape[1]):
            ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
    return ycbcr_img
def ycbcr2rgb(ycbcr_img):
    rgb_img = np.zeros(ycbcr_img.shape, dtype=np.uint8)
    for x in range(ycbcr_img.shape[0]):
        for y in range(ycbcr_img.shape[1]):
            [r, g, b] = ycbcr_img[x,y,:]
            rgb_img[x, y, :] = np.maximum(0, np.minimum(255, np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
    return rgb_img
