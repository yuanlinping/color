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
test_file = '../dataset/test.txt'

image_height = 128
image_width = 256


def get_file_paths (batch_size, file=train_file):
    paths = open(file, 'r').read().splitlines()

    random.shuffle(paths)

    image_paths = [p.split('\t')[0] for p in paths]
    label_paths = [p.split('\t')[1] for p in paths]

    num_batch = len(image_paths) // batch_size

    return {"image_paths": image_paths, "label_paths": label_paths, "num_batch": num_batch}


def image_data_loader(start_index, batch_size, file_paths):
    images = np.zeros((batch_size, image_height, image_width, 1), dtype=np.float32)
    for i in range(batch_size):
        file_path = file_paths[i + start_index]
        feature = np.genfromtxt(file_path, delimiter=",")
        feature = feature.reshape(image_height, image_width, 1)
        images[i] = feature
    # print images.shape
    return images


def label_data_loader(start_index, batch_size, file_paths):
    labels = np.zeros((batch_size, 40, 1024, 3), dtype=np.float32)
    for i in range(batch_size):
        file_path = file_paths[i + start_index]
        lab = imread(file_path, mode='RGB') / 255.
        label = np.reshape(lab, (40, 1024, 3))
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
