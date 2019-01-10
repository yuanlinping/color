import numpy as np

import os
import sys
import glob
import time

from scipy.misc import imread

# train file
data_dir = '../dataset/chart/'
label_dir = '../dataset/legend/'

# im_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
# gt_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))
#
# assert len(im_paths) == len(gt_paths)
#
# train = open('../dataset/stackedBar_40/train.txt', 'w')
# for i in range(len(im_paths)):
# 	print>>train, '{}\t{}'.format(im_paths[i],gt_paths[i])
# train.close()
#
# # test file
# test_data_dir = '../dataset/stackedBar_40/test_chart'
# test_label_dir = '../dataset/stackedBar_40/test_legend'
#
# im_paths = sorted(glob.glob(os.path.join(test_data_dir, '*.png')))
# gt_paths = sorted(glob.glob(os.path.join(test_label_dir, '*.png')))
#
# assert len(im_paths) == len(gt_paths)
#
# test = open('../dataset/stackedBar_40/test.txt', 'w')
# for i in range(len(im_paths)):
# 	print>>test, '{}\t{}'.format(im_paths[i],gt_paths[i])
# test.close()

chartNum = 320
train = open('../dataset/train.txt', 'w')
for i in range(chartNum):
	im_paths = data_dir+"C"+str((i+1))+".png"
	gt_paths = data_dir+"L"+str((i+1))+".png"
	print>>train, '{}\t{}'.format(im_paths,gt_paths)
train.close()

