import numpy as np

import tensorflow as tf

# import cv2

from scipy.misc import imread, imsave, imresize

import os
import sys
import glob
import time
import random

train_file = '../dataset/train.txt'


image_height = 128
image_width = 256
image_channel = 1

label_height = 40
label_width = 1024
label_channel = 3


def get_file_paths (batch_size, file=train_file):
    paths = open(file, 'r').read().splitlines()

    random.shuffle(paths)

    image_paths = [p.split('\t')[0] for p in paths]
    label_paths = [p.split('\t')[1] for p in paths]

    num_batch = len(image_paths) // batch_size

    return {"image_paths": image_paths, "label_paths": label_paths, "num_batch": num_batch}


def image_data_loader(start_index, batch_size, file_paths):
    images = np.zeros((batch_size, image_height, image_width, image_channel), dtype=np.float32)
    for i in range(batch_size):
        file_path = file_paths[i + start_index]
        feature = np.genfromtxt(file_path, delimiter=",")
        # #for line / scatterplot
        # feature[0] = 0.0
        # feature[-1] = 0.0
        # maxV = max(feature)
        # minV = min(feature)
        # diffV = (maxV - minV) * 1.0
        # for v in range(len(feature)):
        #     feature[v] = (feature[v] - minV) / diffV
        feature = feature.reshape(image_height, image_width, image_channel)
        images[i] = feature
    mu = np.mean(images, axis=(0, 1, 2))
    images = images - mu.reshape(1, 1, 1, image_channel)
    # print images.shape
    return images


def label_data_loader(start_index, batch_size, file_paths):
    labels = np.zeros((batch_size, label_height, label_width, label_channel), dtype=np.float32)
    for i in range(batch_size):
        file_path = file_paths[i + start_index]
        lab = imread(file_path, mode='RGB') / 255.
        label = np.reshape(lab, (label_height, label_width, label_channel))
        labels[i] = label
    # print labels.shape
    return labels


# Unit test
# if __name__ == '__main__':
# 	data_dict = data_loader()

# 	images = data_dict['images']
# 	labels = data_dict['labels']
# 	print images.shape, labels.shape

# 	sess = tf.Session()
# 	sess.run(tf.group(tf.global_variables_initializer(),
# 					tf.local_variables_initializer()))

# 	# coordinator for queue runner
# 	coord = tf.train.Coordinator()

# 	# start queue 
# 	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 	batch_im, batch_gt = sess.run([images, labels])
# 	from matplotlib import pyplot as plt
# 	plt.subplot(121)
# 	plt.imshow(batch_im[0,...])
# 	plt.subplot(122)
# 	plt.imshow(batch_gt[0,...])
# 	plt.show()
