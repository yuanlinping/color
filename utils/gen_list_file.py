import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread

# train file
data_dir = '../dataset/chart'
label_dir = '../dataset/legend'

im_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
gt_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))

assert len(im_paths) == len(gt_paths)

train = open('../dataset/train.txt', 'w')
for i in range(len(im_paths)):
	print>>train, '{}\t{}'.format(im_paths[i],gt_paths[i])
train.close()

# test file
test_data_dir = '../dataset/test_chart'
test_label_dir = '../dataset/test_legend'

im_paths = sorted(glob.glob(os.path.join(test_data_dir, '*.png')))
gt_paths = sorted(glob.glob(os.path.join(test_label_dir, '*.png')))

assert len(im_paths) == len(gt_paths)

test = open('../dataset/test.txt', 'w')
for i in range(len(im_paths)):
	print>>test, '{}\t{}'.format(im_paths[i],gt_paths[i])
test.close()
