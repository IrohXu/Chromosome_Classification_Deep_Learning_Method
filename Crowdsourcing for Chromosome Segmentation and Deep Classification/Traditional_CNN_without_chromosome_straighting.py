import os
import numpy as np
np.random.seed(1337)  # for reproducibility
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from numpy import *

file = './single_chromosomes'   # address of dataset

sample_num = 45
items = [i for i in range(1, sample_num + 1)]
import random
test_set = random.sample(items, int(sample_num*0.15))
train_set = items
for i in test_set:
    train_set.remove(i)

def get_file(file_dir, sample_set):
    X = np.empty((0,120))   # X is the image dataset
    y = np.empty(0, dtype='uint8')    # y is the label
    for i in sample_set:
        for f in os.listdir(file_dir + '/' + str(i) + '/'):
            name = f.split(' ')[1].split('.')[0]
            if len(name) == 3:
                if name[2] == 'a' or name[2] == 'b':
                    name = name[0] + name[1]
            elif len(name) == 2:
                if name[1] == 'a' or name[1] == 'b':
                    name = name[0]
            root = file_dir + '/' + str(i) + '/' + f
            # print(name, root)
            pic = Image.open(root)
            pic = pic.convert('L')
            pic = pic.resize((120, 120))   # transform the image to 120*120
            X = np.concatenate([X, np.array(pic)])
            y = np.concatenate([y, np.array([int(name)])])
            # print(np.array(pic))
    return X, y

X_test, y_test = get_file(file, test_set)
batch_set = random.sample(train_set, 5)
X_train, y_train = get_file(file, batch_set)

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

X_train = X_train.reshape(-1, 1,120, 120)/255.
X_test = X_test.reshape(-1, 1,120, 120)/255.
y_train = np_utils.to_categorical(y_train, num_classes=25)
y_test = np_utils.to_categorical(y_test, num_classes=25)

# A way to build your CNN
model = Sequential()

# Conv layer 1 output shape (64, 120, 120)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 120, 120),
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Convolution2D(
    batch_input_shape=(None, 1, 120, 120),
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',     # Padding method
    data_format='channels_first',
))
model.add(Activation('relu'))
# Pooling layer 1 (max pooling) output shape (64, 60, 60)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',    # Padding method
    data_format='channels_first',
))

# Conv layer 2 output shape (32, 60, 60)
model.add(Convolution2D(32, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Convolution2D(32, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (32, 30, 30)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Conv layer 3 output shape (16, 30, 30)
model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
# Pooling layer 3 (max pooling) output shape (16, 15, 15)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Conv layer 4 output shape (8, 15, 15)
model.add(Convolution2D(8, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Convolution2D(8, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
# Pooling layer 4 (max pooling)
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

# Fully connected layer 1 input shape (8 * 7 * 7), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('sigmoid'))

# Fully connected layer 2 input shape (1024), output shape (512)
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('sigmoid'))

# Fully connected layer 3 to shape (25) for 25 classes
model.add(Dense(25))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-4)   # use Adam as optimizer

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1, batch_size=2,)

loss, accuracy = model.evaluate(X_test, y_test)


