import argparse
import numpy as np
from scipy import misc
from os import listdir, makedirs
from os.path import isfile, join, exists
import utils
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default='../DataTrain/RAW', type=str, help="Data input directory")
parser.add_argument("--output_dir", default = '../DataTrain/Images',type=str, help="Data output directory")
args = parser.parse_args()


scale = 3.0
input_size = 33
label_size = 21
pad = int((input_size - label_size) / 2)
stride = 14

if not exists(args.output_dir):
    makedirs(args.output_dir)
if not exists(join(args.output_dir, "input")):
    makedirs(join(args.output_dir, "input"))
if not exists(join(args.output_dir, "label")):
    makedirs(join(args.output_dir, "label"))

count = 1
index = 1
for f in listdir(args.input_dir):
    f = join(args.input_dir, f)
    if not isfile(f):
        continue

    image = np.array(Image.open(f))

    w, h, c = image.shape
    w -= w % 3
    h -= h % 3
    image = image[0:w, 0:h]

    scaled = misc.imresize(image, 1.0/scale, 'bicubic')
    scaled = misc.imresize(scaled, scale/1.0, 'bicubic')

    for i in range(0, h - input_size + 1, stride):
        for j in range(0, w - input_size + 1, stride):
            sub_img = scaled[j : j + input_size, i : i + input_size]
            sub_img_label = image[j + pad : j + pad + label_size, i + pad : i + pad + label_size]
            misc.imsave(join(args.output_dir, "input", str(count) + '.bmp'), sub_img)
            misc.imsave(join(args.output_dir, "label", str(count) + '.bmp'), sub_img_label)

            count += 1
    index += 1
    utils.progressbar(index-2, len(listdir(args.input_dir)), length=40)
