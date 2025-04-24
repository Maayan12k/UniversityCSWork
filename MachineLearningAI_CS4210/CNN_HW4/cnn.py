#-------------------------------------------------------------------------
# AUTHOR: Maayan Israel
# FILENAME: cnn.py
# SPECIFICATION:  Convolutional Neural Network (CNN) for digit classification
# FOR: CS 4210 - Assignment #4
#-------------------------------------------------------------------------

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models

def load_digit_images_from_folder(folder_path, image_size=(32, 32)):
    X = []
    y = []
    for filename in os.listdir(folder_path):
        # Getting the label of the image (it's the first number in the filename)
        label = int(filename[0])

        img = Image.open(os.path.join(folder_path, filename)).convert('L').resize(image_size)

        X.append(np.array(img))
        y.append(label)
    return np.array(X), np.array(y)

train_path = os.path.join("train")
test_path = os.path.join("test")

# Loading the raw images using the provided function. Hint: Use the provided load_digit_images_from_folder function that outputs X_train, Y_train for train_path and as X_test, Y_test for test_path
X_train, Y_train = load_digit_images_from_folder(train_path)
X_test, Y_test = load_digit_images_from_folder(test_path)

# Normalizing the data: convert pixel values from range [0, 255] to [0, 1]. Hint: divide them by 255
X_train = X_train / 255
X_test = X_test / 255

# Reshaping the input images to include the channel dimension: (num_images, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], 32, 32, 1)
X_test  = X_test .reshape(X_test .shape[0], 32, 32, 1)

# Building a CNN model
model = models.Sequential([
    # conv layer: 32 filters, 3x3 kernel, relu, input shape 32×32×1
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
    # max‐pooling 2×2
    layers.MaxPooling2D((2, 2)),
    # flatten to vector
    layers.Flatten(),
    # dense hidden layer
    layers.Dense(64, activation='relu'),
    # output layer for 10 classes
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_test, Y_test)
)

loss, acc = model.evaluate(X_test, Y_test)
print("Test accuracy:", acc)