import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread
import random

# train file
hs_dir = '../dataset/test_hsHis/'
hl_dir = '../dataset/test_hlHis/'

num_sample = 68
start_point = 1

test = open('../dataset/test_1.txt', 'w')
for i in range(num_sample):
	index = i + start_point
	im_paths = hs_dir+"HS" + str(index) + ".csv"
	gt_paths = hl_dir+"HL" + str(index) + ".csv"
	print>> test, '{}\t{}'.format(im_paths, gt_paths)
test.close()

