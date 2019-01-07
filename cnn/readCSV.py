import tensorflow as tf

tfrecords_path = '../dataset/train.tfrecords'

#
# def parser(record):
#     features = {
#         'feats': tf.FixedLenFeature([], tf.string),
#         'label': tf.FixedLenFeature([], tf.string),
#     }
#
#     parsed = tf.parse_single_example(record, features)
#     feats = tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.int32))
#     label = tf.convert_to_tensor(tf.decode_raw(parsed['label'], tf.int32))
#     print(record)
#     print(features)
#     print(feats)
#     print(label)
#     return {'feats': feats, "label": label}
#

filename_queue = tf.train.string_input_producer([tfrecords_path])
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
parsed = tf.parse_single_example(
    serialized_example,
    features={
                'feats': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
    }
)
feats = tf.convert_to_tensor(tf.decode_raw(parsed['feats'], tf.int32))
label = tf.convert_to_tensor(tf.decode_raw(parsed['label'], tf.int32))
print(_)
print(serialized_example)
print(feats)
print(label)
# return {'feats': feats, "label": label}

# img = tf.decode_raw(features['img_raw'], tf.uint8)
# img = tf.reshape(img, [image_size, image_size, 1])
# label = tf.cast(features['label'], tf.int32)

# # print(parser)
# dataset = (
#     tf.data.TFRecordDataset(tfrecords_path)
#         .map(parser)
#         .batch(4)
# )
# raw_dataset = (tf.data.TFRecordDataset(tfrecords_path))
# print(raw_dataset)
# print(dataset)
#
# iterator = dataset.make_one_shot_iterator()
#
# batch_feats, batch_labels = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(1):
#         # feats = sess.run(batch_feats)
#         labels = sess.run(batch_labels)
#         # print(feats)

# dataset = tf.data.Dataset.range(10)
#
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
# with tf.Session() as sess:
#     for i in range(10):
#       value = sess.run(next_element)
#       print(value)

