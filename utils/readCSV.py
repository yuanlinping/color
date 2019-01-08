import numpy as np

import tensorflow as tf


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


train_label = "../dataset/label.csv"
train_feature = "../dataset/example.csv"

label = np.genfromtxt(train_label, delimiter=",")
feature = np.genfromtxt(train_feature, delimiter=",")

label, feature = unison_shuffled_copies(label,feature)

print(label)
print(feature)

label = label.reshape(len(label),2,2)
# feature = feature.reshape(2,2,len(feature))

print(1 << 10)

print(label)
# print(feature)

label = tf.convert_to_tensor(label, dtype=tf.int32)
feature = tf.convert_to_tensor(feature, dtype=tf.int32)
