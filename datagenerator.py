#encoding=utf-8
import numpy as np
import cv2
import os
import glob

class ImageDataGenerator:
    def __init__(self,image_list,label_list,horizontal_flip=False,shuffle=False,mean=np.array([104., 117., 124.]),scale_size=(227, 227),nb_classes=32):

        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        self.labels=label_list
        self.images=["picture/"+i for i in image_list]
        self.data_size=len(self.labels)

        #self.test_read(class_list)

        if self.shuffle:
            self.shuffle_data()
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        # python 3....
        #images = self.images.copy()
        #labels = self.labels.copy()
        images = self.images
        labels = self.labels
        self.images = []
        self.labels = []

        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0

        if self.shuffle:
            self.shuffle_data()

    def next_batch(self, batch_size):
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]

        self.pointer += batch_size

        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
            img = cv2.imread(paths[i])
            #rescale image
            img = cv2.resize(img, (self.scale_size[0], self.scale_size[0]))
            img = img.astype(np.float32)
            #subtract mean
            img -= self.mean
            images[i] = img
        # Expand labels to one hot encoding
        one_hot_labels = np.zeros((batch_size, self.n_classes))
        for i in range(len(labels)):
            one_hot_labels[i][labels[i]] = 1
        #return array of images and labels
        return images, one_hot_labels



