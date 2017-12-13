import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

'''
writer=tf.python_io.TFRecordWriter("train.tfrecords")
file=open("train.txt","r")
for line in file.readlines():
    line=line.rstrip("\n")
    line=line.split("\t")
    name="picture/"+line[1]
    print(line[1])
    label=int(line[-1])
    image=Image.open(name)
    image=image.resize((227,227))
    img_raw=image.tobytes()
    example=tf.train.Example(features=tf.train.Features(feature={
        'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))

    }))
    writer.write(example.SerializeToString())


writer.close()
'''
filename_queue=tf.train.string_input_producer(["train.tfrecords"])
reader=tf.TFRecordReader()
_,serialized_example=reader.read(filename_queue)
features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
img = tf.decode_raw(features['img_raw'], tf.uint8)
img = tf.reshape(img, [227, 227, 3])
img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
label = tf.cast(features['label'], tf.int32)

img_batch,label_batch=tf.train.shuffle_batch([img,label],batch_size=25,capacity=2000,min_after_dequeue=1000)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    threads=tf.train.start_queue_runners(sess=sess)
    for i in range(2):
        val,l=sess.run([img_batch,label_batch])
        print(val.shape,l)


