import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread

# train file
import random

data_dir = '../dataset/chart/'
label_dir = '../dataset/legend/'


im_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
gt_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))

index_array = range(len(im_paths))

random.shuffle(index_array)

start_point = 3841
range_1 = 864
range_2 = len(im_paths) - range_1


train = open('../dataset/train.txt', 'w')
for i in range(range_1):
	index = index_array[i] + start_point
	im_path = data_dir + "C" + str(index) + ".png"
	gt_path = label_dir + "L" + str(index) + ".png"
	print>>train, '{}\t{}'.format(im_path, gt_path)
train.close()


test = open('../dataset/test.txt', 'w')
for i in range(range_2):
	index = index_array[i+range_1] + start_point
	im_path = data_dir + "C" + str(index) + ".png"
	gt_path = label_dir + "L" + str(index) + ".png"
	print>>test, '{}\t{}'.format(im_path, gt_path)
test.close()