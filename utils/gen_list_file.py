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

index_array = random.shuffle(range(len(im_paths)))

train = open('../dataset/train.txt', 'w')
for i in range(300):
	index = index_array[i] + 1
	im_path = data_dir + "C" + str(index) + ".png"
	gt_path = label_dir + "L" + str(index) + ".png"
	print>>train, '{}\t{}'.format(im_path, gt_path)
train.close()


test = open('../dataset/test.txt', 'w')
for i in range(20):
	index = index_array[i+300] + 1
	im_path = data_dir + "C" + str(index) + ".png"
	gt_path = label_dir + "L" + str(index) + ".png"
	print>>test, '{}\t{}'.format(im_path, gt_path)
test.close()
