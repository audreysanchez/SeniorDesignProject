import tensorflow as tf
import os
import random
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from keras import preprocessing
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

class_names = ['DRONE', 'NO_DRONE']

width = 150
height = 150

def load_images(base_path):
    images = []
    path = os.path.join(base_path, '*.jpeg')
    for image_path in glob(path):
        image = preprocessing.image.load_img(image_path, target_size=(width, height))
        x = preprocessing.image.img_to_array(image)
        images.append(x)
    return images

training_drone = load_images('./drone')
training_noDrone = load_images('./NoDrone')

plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(random.choice(training_drone))
    plt.imshow(image)
    plt.axis('off')
    plt.title('{} image'.format(class_names[0]))

# show the plot
plt.show()

plt.figure(figsize=(12,8))

for i in range(5):
    plt.subplot(1, 5, i+1)
    image = preprocessing.image.array_to_img(random.choice(training_noDrone))
    plt.imshow(image)

    plt.axis('off')
    plt.title('{} image'.format(class_names[1]))

# show the plot
plt.show()

#prepare images as tensors

X_drone = np.array(training_drone)
X_noDrone = np.array(training_noDrone)

print(X_drone.shape)
print(X_noDrone.shape)

X = np.concatenate((X_drone, X_noDrone), axis=0)
X = X / 255
X.shape

y_drone = [0 for item in enumerate(X_drone)]
y_noDrone = [1 for item in enumerate(X_noDrone)]

y = np.concatenate((y_drone, y_noDrone), axis=0)

y = to_categorical(y, num_classes=len(class_names))

print(y.shape)

#convolutional neural network configuration

# default parameters
conv_1 = 16
conv_1_drop = 0.2
conv_2 = 32
conv_2_drop = 0.2
dense_1_n = 1024
dense_1_drop = 0.2
dense_2_n = 512
dense_2_drop = 0.2
lr = 0.001

epochs = 30
batch_size = 32
color_channels = 3

def build_model(conv_1_drop=conv_1_drop, conv_2_drop=conv_2_drop,
        dense_1_n=dense_1_n, dense_1_drop=dense_1_drop,
        dense_2_n=dense_2_n, dense_2_drop=dense_2_drop,
        lr=lr):
    model = Sequential()
    model.add(Convolution2D(conv_1, (3, 3), input_shape=(width, height, color_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_1_drop))
    model.add(Convolution2D(conv_2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(conv_2_drop))
    model.add(Flatten())
    model.add(Dense(dense_1_n, activation='relu'))
    model.add(Dropout(dense_1_drop))
    model.add(Dense(dense_2_n, activation='relu'))
    model.add(Dropout(dense_2_drop))
    model.add(Dense(len(class_names), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr),
    metrics=['accuracy'])
    
    return model

np.random.seed(1) # for reproducibility

# model with base parameters
model = build_model()

epochs = 20

# "X" is the array of images, "y" is the type of the images
model.fit(X,y, epochs=epochs) #end training model

model.save('drone_detector.h5') #save training model