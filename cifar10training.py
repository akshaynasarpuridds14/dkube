#!/usr/bin/env python
# coding: utf-8

# Import required packages

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os
import shutil

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten


# Environment Variables and Directory Structure

BATCH_SIZE = int(os.getenv('BATCHSIZE', 64))
EPOCHS = int(os.getenv('EPOCHS', 10))
NUM_CLASSES = 10
DATA_DIR='/opt/dkube/input'
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('values'):
    os.makedirs('values')
MODEL_DIR='/opt/dkube/output'


path=DATA_DIR+'/train'
path_labels=DATA_DIR+'/trainLabels.csv'


train_dir = os.listdir(DATA_DIR + '/train')
train_dir_len = len(train_dir)
print("Length:\t", train_dir_len)


train_labels = pd.read_csv(path_labels)
train_images = pd.DataFrame(columns=['id', 'label', 'path'], dtype=str)


train_root = path

for i in range(0, train_dir_len):
    path1 = os.path.join(train_root, str(i+1) + ".png")
    if os.path.exists(path1):
        train_images = train_images.append([{
            'id': train_labels['id'].iloc[i],
            'label': train_labels['label'].iloc[i],
            'path': str(i+1) + '.png'
        }])


display_groupby = train_images.groupby(['label']).count()


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for name in class_names:
    index = class_names.index(name)
    train_images.loc[train_images['label'] == name, 'label'] = str(index)

display_groupby = train_images.groupby(['label']).count()

# Image Data Generator

data_generator = ImageDataGenerator(rescale=1/255.,
                                   validation_split=0.2,
                                   horizontal_flip=True)

train_generator = data_generator.flow_from_dataframe(dataframe=train_images, 
                                                    directory=train_root,
                                                    x_col='path',
                                                     y_col='label',
                                                     subset='training',
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(32,32),
                                                     class_mode='categorical')

validation_generator = data_generator.flow_from_dataframe(dataframe=train_images,
                                                         directory=train_root,
                                                         x_col='path',
                                                         y_col='label',
                                                         subset='validation',
                                                         batch_size=BATCH_SIZE,
                                                         shuffle=True,
                                                         target_size=(32,32),
                                                         class_mode='categorical')


# Modeling


model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=(32,32,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=3, activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.001), metrics=['accuracy'])
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.6, min_denta=0.00001)
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)

history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, callbacks=[reduce_lr, es])

version='2'
model.save(MODEL_DIR+'/'+version)


