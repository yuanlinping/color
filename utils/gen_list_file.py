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
range_1 = 300
range_2 = num_sample - range_1
start_point = 1

train = open('../dataset/train.txt', 'w')
for i in range(range_1):
	index = index_array[i] + start_point
	im_paths = data_dir+"C" + str(index) + ".csv"
	gt_paths = label_dir+"L" + str(index) + ".png"
	print>> train, '{}\t{}'.format(im_paths, gt_paths)
train.close()

test = open('../dataset/test.txt', 'w')
for i in range(range_2):
	index = index_array[i+range_1] + start_point
	im_paths = data_dir+"C" + str(index) + ".csv"
	gt_paths = label_dir+"L" + str(index) + ".png"
	print>> test, '{}\t{}'.format(im_paths, gt_paths)
test.close()

