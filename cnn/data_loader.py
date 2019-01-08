import numpy as np

import tensorflow as tf

import cv2

from scipy.misc import imread, imsave, imresize

import os
import sys
import glob
import time

import random

# train_file = '../dataset/train.txt'
# test_file = '../dataset/test.txt'

train_label = "../dataset/labelFromJS.csv"
train_feature = "../dataset/chart.csv"
# train_label = "../dataset/label.csv"
# train_feature = "../dataset/example.csv"


def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]


def data_loader(batch_size=1,type="train",resize=None):
	label = []
	feature = []

	if(type=="train"):
		label = np.genfromtxt(train_label, delimiter=",")
		feature = np.genfromtxt(train_feature, delimiter=",")

	label, feature = unison_shuffled_copies(label, feature)

	label = label.reshape(len(label), 4, 15, 1)
	feature = feature.reshape(len(feature), 1 << 7, 1 << 8, 1)

	labels = tf.convert_to_tensor(label, dtype=tf.float32)
	images = tf.convert_to_tensor(feature, dtype=tf.float32)


	# number of batches
	num_batch = len(feature) // batch_size

	return {'images': images, 'labels': labels, 'num_batch': num_batch}



# def data_loader(batch_size=1, file=train_file, resize=None):
# 	"""
# 	Read pair of training set
# 	Use fixed input size: [512x1024x3], gt size: [40x1024x3]
# 	"""
#
# 	paths = open(file, 'r').read().splitlines()
#
# 	random.shuffle(paths)
#
# 	image_paths = [p.split('\t')[0] for p in paths]
# 	label_paths = [p.split('\t')[1] for p in paths]
#
# 	# create batch input
# 	# convert to tensor list
# 	img_list  = tf.convert_to_tensor(image_paths, dtype=tf.string)
# 	lab_list = tf.convert_to_tensor(label_paths, dtype=tf.string)
#
# 	with tf.Session() as sess:
# 		sess.run([img_list,lab_list])
# 		print(img_list.eval())
# 		print(lab_list.eval())
#
# 	# create data queue
# 	data_queue = tf.train.slice_input_producer([img_list, lab_list],
# 		shuffle=False, capacity=batch_size*128)
#
# 	# decode image
# 	image = tf.image.decode_png(tf.read_file(data_queue[0]), channels=3)
# 	label = tf.image.decode_png(tf.read_file(data_queue[1]), channels=3)
#
# 	# resize to define image shape
# 	if resize is None:
# 		image = tf.reshape(image, [512, 1024, 3])
# 		label = tf.reshape(label, [40, 1024, 3])
# 	else:
# 		image = tf.image.resize_images(image, resize)
# 		label = tf.image.resize_images(label, resize)
#
# 	# convert to float data type
# 	image = tf.cast(image, dtype=tf.float32)
# 	label  = tf.cast(label, dtype=tf.float32)
#
# 	# data pre-processing, normalize
# 	image = tf.divide(image, tf.constant(255.0))
# 	label = tf.divide(label, tf.constant(255.0))
#
# 	# one-hot label, convert to one-hot label during loss computation
# 	# label =
#
# 	# create batch data
# 	images, labels = tf.train.shuffle_batch([image, label],
# 		batch_size=batch_size, num_threads=1,
# 		capacity=batch_size*128, min_after_dequeue=batch_size*32)
#
# 	# number of batches
# 	num_batch = len(image_paths) // batch_size
#
# 	return {'images':images, 'labels':labels, 'num_batch':num_batch}


