from __future__ import absolute_import, division, print_function, unicode_literals
from keras import layers
from keras import models
from keras.applications import VGG16
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.applications.vgg19 import VGG19
from datetime import datetime
import os
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import IPython.display as display
from PIL import Image
import numpy as np
import pathlib
import matplotlib.pyplot as plt
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.__version__

from keras.models import Sequential, Model
from keras import backend as k

TRAIN_PATH = 'Train'
TEST_PATH = 'Test'

BATCH_SIZE = 16
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 4


data_dir = pathlib.Path(TRAIN_PATH)
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])


def data_generator(dir_path):
    data_dir = pathlib.Path(dir_path)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    # print(CLASS_NAMES)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                      brightness_range=[0.7, 1.0],)
    data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   target_size=(
                                                       IMG_HEIGHT, IMG_WIDTH),
                                                   classes=list(CLASS_NAMES))
    return data_gen, STEPS_PER_EPOCH


def train_valid_generator(dir_path):
    data_dir = pathlib.Path(dir_path)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    # print(CLASS_NAMES)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                      validation_split=0.2,
                                                                      brightness_range=[0.7, 1.0],)

    train_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(
                                                        IMG_HEIGHT, IMG_WIDTH),
                                                    classes=list(CLASS_NAMES),
                                                    subset='training')

    valid_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    target_size=(
                                                        IMG_HEIGHT, IMG_WIDTH),
                                                    classes=list(CLASS_NAMES),
                                                    subset='validation')

    return train_gen, valid_gen


train_data_gen, valid_data_gen = train_valid_generator(TRAIN_PATH)
test_data_gen, STEPS_TEST = data_generator(TEST_PATH)

#Load the VGG model
vgg_conv = VGG19(weights='imagenet', include_top=False,
                 input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Freeze the layers except the last 4 layers
for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

# Check the trainable status of the individual layers
for layer in vgg_conv.layers:
    print(layer, layer.trainable)


# Create the model
model = models.Sequential()

# Add the vgg convolutional base model
model.add(vgg_conv)

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4, activation='softmax'))

# Show a summary of the model. Check the number of trainable parameters
print(model.summary())

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Train the model
history = model.fit_generator(
    generator=train_data_gen,
    steps_per_epoch=train_data_gen.samples//BATCH_SIZE,
    validation_data=valid_data_gen,
    validation_steps=valid_data_gen.samples//BATCH_SIZE,
    epochs=EPOCHS)

now = datetime.now()
model.save('models/model_' + str(now) + '.h5')

# Plotting the model
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
