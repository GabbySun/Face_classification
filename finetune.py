#!/usr/bin/env python
#encoding=utf-8
import os
import time
import numpy as np
import tensorflow as tf
from datetime import datetime
from alexnet import AlexNet
from datagenerator import ImageDataGenerator
from graph_construction import graph

train_file = "train.txt"
val_file = "val.txt"
filewriter_path = './data/filewriter'
checkpoint_path = './data/checkpoint'


#parameters
learning_rate=0.01
num_epochs=20
batch_size=25
dropout_rate = 0.5
num_classes = 32
train_layers = ['fc8','fc7']
display_step = 1
top_N = 1
file1=open(train_file,"r")
file2=open(val_file,"r")
train_image=[]
train_label=[]
val_image=[]
val_truth=[]
train_index=[]
val_index=[]
for line in file1.readlines():
    line=line.rstrip("\n")
    line=line.split("\t")
    train_index.append(int(line[0]))
    train_image.append(line[1])
    train_label.append(int(line[-1]))
for line in file2.readlines():
    line = line.rstrip("\n")
    line = line.split("\t")
    val_index.append(int(line[0]))
    val_image.append(line[1])
    val_truth.append(int(line[-1]))



restore_checkpoint =""
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)

model = AlexNet(x, keep_prob, num_classes, train_layers)

score = model.fc8

var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]  #获取参数只要需要训练的参数

smoothness=0
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score, labels=y))+smoothness

with tf.name_scope("train"):
    gradients = tf.gradients(loss, var_list)  #导数
    gradients = list(zip(gradients, var_list))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)


for gradient, var in gradients:
    tf.summary.histogram(var.name +'/gradient', gradient)

for var in var_list:
    tf.summary.histogram(var.name, var)
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("accuracy"):
    labels = tf.argmax(y, 1)
    topFiver = tf.nn.in_top_k(score, labels, top_N)
    accuracy = tf.reduce_mean(tf.cast(topFiver, tf.float32))
    prediction=tf.argmax(score,1)


tf.summary.scalar('accuracy', accuracy)
merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter(filewriter_path)

saver = tf.train.Saver()

train_generator = ImageDataGenerator(
    train_image,train_label, horizontal_flip=False, shuffle=False)
val_generator = ImageDataGenerator(val_image,val_truth, shuffle=False)


train_batches_per_epoch = np.floor(train_generator.data_size /
                                   batch_size).astype(np.int16)
val_batches_per_epoch = np.floor(val_generator.data_size /
                                 batch_size).astype(np.int16)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer.add_graph(sess.graph)

    if restore_checkpoint == '':
        model.load_initial_weights(sess)
    else:
        saver.restore(sess, restore_checkpoint)


    print("{} Start training...".format(datetime.now()))


    for epoch in range(num_epochs):
        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        val_label=[]
        step = 1
        train_count=0
        train_acc=0

        while step <= train_batches_per_epoch:

            start_time = time.time()
            batch_xs, batch_ys = train_generator.next_batch(batch_size)
            sess.run(train_op,feed_dict={x: batch_xs,y: batch_ys,keep_prob: dropout_rate})
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            train_acc += acc
            train_count += 1
            duration = time.time() - start_time

            if step % display_step == 0:
                s = sess.run(merged_summary,feed_dict={x: batch_xs, y: batch_ys,keep_prob: 1.})
                writer.add_summary(s, epoch * train_batches_per_epoch + step)
            if step % 10 == 0:
                print("[INFO] {} pics has trained. time using {}".format(step*batch_size,duration))
            step += 1
        train_acc /= train_count
        print("train accuracy:{}".format(train_acc))

        print("{} Start validation".format(datetime.now()))
        test_acc = 0.
        test_count = 0

        for _ in range(val_batches_per_epoch):
            batch_tx, batch_ty = val_generator.next_batch(batch_size)
            val_score= sess.run(prediction, feed_dict={x: batch_tx,y: batch_ty,keep_prob: 1.})
            val_label.extend(val_score)
            print("Validation score = {}".format(val_score))
            acc=sess.run(accuracy, feed_dict={x: batch_tx,y: batch_ty,keep_prob: 1.})
            label=sess.run(labels,feed_dict={x:batch_tx,y:batch_ty,keep_prob:1})
            print("labels={}".format(label))
            test_acc += acc
            test_count += 1
        test_acc /= test_count
        print("Validation score = {} {}".format(datetime.now(), test_acc))

        '''
        smoothness computation:
        '''
        val_num=len(val_label)
        val_graph_index=val_index[:val_num]
        construction_graph=graph(train_index,train_label,val_graph_index,val_label)
        smoothness=construction_graph.smoothness*0.01


        val_generator.reset_pointer()
        train_generator.reset_pointer()


        '''
        print("{} Saving checkpoint of model...".format(datetime.now()))

        checkpoint_name = os.path.join(
            checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)
        print("{} Model checkpoint saved at {}".format(datetime.now(),checkpoint_name))
        '''