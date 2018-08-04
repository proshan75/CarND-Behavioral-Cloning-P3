# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:00:02 2018

@author: Prashant Borse
"""

import os
import csv

# Read the driving_log.csv file, it contains camera images from vehicle's
# left, right and center. Also, it has records for steering angle, throttle,
# braking and speed value.
samples = []
with open('../../Data_train/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

# generator method to loop throgh the samples read from the csv file.
# Each line in the sample is parsed for image name and steering value.
# As the data getting processed it is shuffled to help generalizing and avoid
# overfitting. The data is processed per input batch size.
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) # shuffle input data lines parsed from csv file
        for offset in range(0, num_samples, batch_size):
            # get data of input batch size
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # process batches for extracting images and steering value
            for batch_sample in batch_samples:
                # retrieve steering angle value from the fourth column
                angle = float(batch_sample[3])
                # first three columns contain images in the batch sample,
                # processing each image column.
                for index in range(3):
                    # retrieve image name
                    name = '../../Data_train/IMG/'+batch_sample[index].split('/')[-1]
                    # using cv2 library read the image. Note that by default
                    # it outputs the image in BGR format
                    image = cv2.imread(name)
                    # convert the image to RGB format
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # append the image to images array
                    images.append(image)

                    # steering data captured in the csv file is applicable
                    # for center image (i.e. index = 0).
                    if index == 0:
                        angles.append(angle)
                    # in case of left image (i.e. index = 1) small value (0.18)
                    # is added to angle in order turn it right
                    elif index == 1:
                        angles.append(angle+0.22)
                    # similarly for right image (i.e. index = 2), the small value
                    # is subtracted to turn the steering to left
                    elif index == 2:
                        angles.append(angle-0.22)


            # generate arrays for X_train and y_train
            X_train = np.array(images)
            y_train = np.array(angles)
            #print('len(X_train): ', len(X_train))
            # shuffle the data before returning
            # also yield until the returned data gets processed, helping memory
            # utilization for processing 20k+ images.
            yield sklearn.utils.shuffle(X_train, y_train)


from sklearn.model_selection import train_test_split
# split the parsed data into training (80%) and validation data (20%)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 160, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, LeakyReLU
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
import matplotlib.pyplot as plt

# building Keras sequential model
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation
# performing normalization
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

# preprocessing image by cropping 75 pixels from top and 20 pixels from bottom
model.add(Cropping2D(cropping=((75,20),(0,0))))
# ----------------------------------------------------
# improvised Nvidia autonomous neural network pipeline
# ----------------------------------------------------
# First layer defined as Convolution2D with 24 filters and 3x3 strides,
# the activation funtion is set to RELU
# Brief explaination of convolution on images is given below:
# http://machinelearninguru.com/computer_vision/basics/convolution/image_convolution_1.html
model.add(Convolution2D(24,3,3, subsample=(2,2), activation="relu"))
# down sampling performed using MaxPooling2D
# simple explaination at:
# https://www.quora.com/What-is-max-pooling-in-convolutional-neural-networks
model.add(MaxPooling2D())
model.add(Convolution2D(48,1,1, subsample=(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,1,1, activation="relu"))
# flatten all layers in preparation for feature detection
# simple explaination at:
# https://www.quora.com/What-is-the-meaning-of-flattening-step-in-a-convolutional-neural-network
model.add(Flatten())
# To avoid overfitting, the dropout is used with 25% neurons dropped randomly
# simple explaination at:
# https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning-And-why-is-it-claimed-to-be-an-effective-trick-to-improve-your-network
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dense(10))
model.add(Dense(1))

# configuring model for training purposeself.
# mean squared error (mse) is used as a loss function.
# Ad adam gradient optimization function is set as a optimier.
model.compile(loss='mse', optimizer='adam')
# as we have generator method setup, here we are using fit_generator to train the
# model with data retrieved and training peformed in parallal
history_object = model.fit_generator(train_generator, validation_data=validation_generator,nb_val_samples=len(validation_samples)*3,  nb_epoch=5, verbose = 1, samples_per_epoch= len(train_samples)*3)

# saving trainted model
model.save('model.h5')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

exit()
