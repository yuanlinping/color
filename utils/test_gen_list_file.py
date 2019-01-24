import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread
import random

# train file
data_dir = '../dataset/test_chart/'
label_dir = '../dataset/test_legend/'

num_sample = 3
start_point = 1

test = open('../dataset/test_1.txt', 'w')
for i in range(num_sample):
	index = i + start_point
	im_paths = data_dir+"C" + str(index) + ".csv"
	gt_paths = label_dir+"L" + str(index) + ".png"
	print>> test, '{}\t{}'.format(im_paths, gt_paths)
test.close()

