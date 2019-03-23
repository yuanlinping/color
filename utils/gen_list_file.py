import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread
import random

# train file
hs_data_dir = '../dataset/hsHis/'
hl_data_dir = '../dataset/hlHis/'
label_dir = '../dataset/legend/'

num_sample = 29040
index_array = range(num_sample)

random.shuffle(index_array)
range_1 = 29040 - 2940
range_2 = num_sample - range_1
start_point = 1
#gray_start = 1440 + start_point
#gray_end = 1679 + start_point

train = open('../dataset/train.txt', 'w')
for i in range(range_1):
	index = index_array[i] + start_point
	#if(index >= gray_start and index <= gray_end):
	#	continue
	hs_paths = hs_data_dir + "HS" + str(index) + ".csv"
	hl_paths = hl_data_dir + "HL" + str(index) + ".csv"
	gt_paths = label_dir + "L" + str(index) + ".png"
	print>> train, '{}\t{}\t{}'.format(hs_paths, hl_paths, gt_paths)
train.close()

test = open('../dataset/test.txt', 'w')
for i in range(range_2):
	index = index_array[i+range_1] + start_point
	#if (index >= gray_start and index <= gray_end):
	#	continue
	hs_paths = hs_data_dir + "HS" + str(index) + ".csv"
	hl_paths = hl_data_dir + "HL" + str(index) + ".csv"
	gt_paths = label_dir+"L" + str(index) + ".png"
	print>> test, '{}\t{}\t{}'.format(hs_paths, hl_paths, gt_paths)
test.close()

