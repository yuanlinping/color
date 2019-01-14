import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread
import random

# train file
data_dir = '../dataset/chartHis_32/'
label_dir = '../dataset/legend/'

num_sample = 320
index_array = range(num_sample)

random.shuffle(index_array)

train = open('../dataset/train.txt', 'w')
for i in range(300):
	index = index_array[i] + 1
	im_paths = data_dir+"C" + str(index) + ".csv"
	gt_paths = label_dir+"L" + str(index) + ".png"
	print>> train, '{}\t{}'.format(im_paths, gt_paths)
train.close()

test = open('../dataset/test.txt', 'w')
for i in range(20):
	index = index_array[i+300] + 1
	im_paths = data_dir+"C" + str(index) + ".csv"
	gt_paths = label_dir+"L" + str(index) + ".png"
	print>> test, '{}\t{}'.format(im_paths, gt_paths)
test.close()

