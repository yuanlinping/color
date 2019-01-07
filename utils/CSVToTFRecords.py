import tensorflow as tf
import numpy as np
import pandas

#
# USAGE: $ python3 csv-to-tfrecords.py data.csv data.tfrecords
#

train_example_file = '../dataset/example.csv'
train_label_file = '../dataset/label.csv'

outfile = '../dataset/train.tfrecords'

label_csv = pandas.read_csv(train_label_file, header=None).values
example_csv = pandas.read_csv(train_example_file, header=None).values

with tf.python_io.TFRecordWriter(outfile) as writer:
    for index in range(4):
        label_row = label_csv[index]
        example_row = example_csv[index]
        print("label and example rows")
        print(label_row)
        print(example_row)

        label = np.array([int(label) for label in label_row]).tostring()
        feats = np.array([int(feat) for feat in example_row]).tostring()

        print(label)
        print(feats)

        example = tf.train.Example()
        example.features.feature["feats"].bytes_list.value.append(feats)
        example.features.feature["label"].bytes_list.value.append(label)
        writer.write(example.SerializeToString())

