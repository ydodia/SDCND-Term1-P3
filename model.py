"""
Self Driving Car ND - Term 1 - Udacity
by - YKD
Aug. 9, 2017

"""

import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
#
import os
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

# Track 1
lines = []
with open('./linux_sim/data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# my generator function outputs X & y array of size batch_size
# since I use data augmentation, the batch_size is divided by four;
# i.e. for each line, there are 2 additional camera images (L & R),
# as well as the mirrored image, for a total of 4.
def my_generator(samples, batch_size=128):
    n_samples = len(samples)
    current_path = './linux_sim/data2/IMG/'
    while 1:
        shuffle(samples)
        for offset in range(0, n_samples, batch_size):
            batch_samples = samples[offset : offset + int(batch_size/4)]
            images = []
            angles = []
            for line in batch_samples:
                img_left_filename = line[1].split('/')[-1]
                img_center_filename = line[0].split('/')[-1]
                img_right_filename = line[2].split('/')[-1]
                img_left = cv2.imread(current_path + img_left_filename)
                img_center = cv2.imread(current_path + img_center_filename)
                img_right = cv2.imread(current_path + img_right_filename)
                images.extend([img_left, img_center, cv2.flip(img_center,1), img_right])
                #
                correction = 0.11
                measurement_center = float(line[3])
                measurement_left = measurement_center + correction
                measurement_right = measurement_center - correction
                angles.extend([measurement_left, measurement_center, -1.0 * measurement_center, measurement_right])
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# build Keras model
# I crop the image, removing the top 70 px and bottom 25 px then normalize.
# The model used is the one from NVIDIA, consisting of:
#   2D-Convolutional layers for first five layers;
#   Flatten layer;
#   Three fully connected layers of decreasing sizes;
#   Finally, a fully-connected layer output the single value for drive angle.
w = 320
h = 160
batch_size = 32
samples_per_epoch = int(len(train_samples) / batch_size) * batch_size
print("Training data has size: {0}; Validation data is: {1}".format(len(train_samples), len(validation_samples)))
train_gen = my_generator(train_samples, batch_size)
validation_gen = my_generator(validation_samples, batch_size)

model = Sequential()
model.add(Cropping2D( cropping=((70, 25), (0,0)), input_shape=(h,w,3)))
model.add(Lambda(lambda  x: (x / 127.5) - 1.0))

model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, samples_per_epoch=samples_per_epoch, validation_data=validation_gen,
                    nb_val_samples=len(validation_samples), nb_epoch=10)

model.save('model.h5')





