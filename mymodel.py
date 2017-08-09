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



# Track 1
lines = []
with open('./linux_sim/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    img_center_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './linux_sim/data/IMG/' + filename
    image = cv2.imread(current_path)
    #image = image[:, 80:240, :]
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(-1.0 * measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
# build Keras model
model = Sequential()
model.add(Cropping2D( cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda  x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))




model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')





