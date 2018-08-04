# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 22:00:02 2018

@author: Prashant Borse
"""

import os
import csv

samples = []
with open('../Data_train/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    name = '../Data_train/IMG/'+batch_sample[i].split('/')[-1]
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    angle = float(batch_sample[3])
                    images.append(image)
    
                    measurement = float(line[3])
                    if i == 0:
                        angles.append(angle)
                    elif i == 1:
                        angles.append(angle+0.20)
                    elif i == 2:
                        angles.append(angle-0.20)


            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            #print('len(X_train): ', len(X_train))
            yield sklearn.utils.shuffle(X_train, y_train)


from sklearn.model_selection import train_test_split
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

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch),
        output_shape=(row, col, ch)))

model.add(Cropping2D(cropping=((75,20),(0,0))))
model.add(Convolution2D(24,3,3, subsample=(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48,1,1, subsample=(2,2), activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64,1,1, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(LeakyReLU())
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, validation_data=validation_generator,nb_val_samples=len(validation_samples)*3,  nb_epoch=5, verbose = 1, samples_per_epoch= len(train_samples)*3)

model.save('model_Nvidea_network_augmented_generator_org_data_prj2.h5')
#model.save('model_Nvidea_network_augmented_cropping_my_data.h5')

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