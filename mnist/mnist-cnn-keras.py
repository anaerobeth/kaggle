"""
Kaggle Competition: Digit Recognizer
https://www.kaggle.com/c/digit-recognizer/data

Data: gray-scale images of hand-drawn digits, from zero through nine
https://www.kaggle.com/c/digit-recognizer/data

Algorithms Used: CNN
Submissions and Public Score:
1-CNN+4XData+Keras - 0.99628

References:
- https://www.kaggle.com/dhimananubhav/mnist-99-74-with-convoluted-nn-and-keras
"""

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib

# Workaround for MacOS/conda setup
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras import models, regularizers, layers, optimizers, losses, metrics
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils, to_categorical

from keras.datasets import mnist

# Load Kaggle augmented dataset (blur/fade/zap/noise effects applied)
if os.path.isfile('train-aug.p'):
    train = pickle.load(open('train-aug.p', 'rb'))
else:
    train = pd.read_csv('data/train-augmented.csv')
    pickle.dump(train, open('train-aug.p', 'wb'))

if os.path.isfile('test.p'):
    test = pickle.load(open('test.p', 'rb'))
else:
    test = pd.read_csv('data/test.csv')
    pickle.dump(test, open('test.p', 'wb'))

train_labels = train['label']
train = train.drop('label', axis=1)

# Reshape to 28 x 28
train_reshaped = np.array(train).reshape(168000, 28, 28)
test_reshaped = np.array(test).reshape(28000, 28, 28)

# print('Training Image: ', train_labels[100])
# plt.imshow(train_reshaped[100], cmap=plt.cm.binary)
# plt.show()

# print('Test Image: ')
# plt.imshow(test_reshaped[100], cmap=plt.cm.binary)
# plt.show()

train_scaled = train_reshaped.reshape(168000, 28, 28, 1) / 255.0
test_scaled = test_reshaped.reshape(28000, 28, 28, 1) / 255.0

# One-hot encode labels
train_labels = to_categorical(train_labels)

# Load additional images from Keras
(train_k, train_labels_k), (test_k, test_labels_k) = mnist.load_data()
train_k_scaled = train_k.reshape(60000, 28, 28, 1) / 255.0
test_k_scaled = test_k.reshape(10000, 28, 28, 1) / 255.0
train_labels_k = to_categorical(train_labels_k)
test_labels_k = to_categorical(test_labels_k)

# Combine Kaggle and Keras datasets
train_images = np.concatenate((train_k_scaled, train_scaled), axis=0)
train_labels = np.concatenate((train_labels_k, train_labels), axis=0)
print('Total training images: ', train_images.shape)
print('Total labels: ', train_labels.shape)

# Initial model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
# Total params: 93,322
# Trainable params: 93,322
# Non-trainable params: 0

# Fit and evaluate
num_epochs = 30
batch_size = 2048
print('fitting model')
model.fit(train_images, train_labels, epochs=num_epochs, batch_size=batch_size)
test_loss, test_acc = model.evaluate(test_k_scaled, test_labels_k)
print('-'*20)
print('Accuracy on test data: ', test_acc)

# Model Stats
# Epoch 30/30
# 228000/228000 [==============================] - 132s 577us/step - loss: 0.0203 - acc: 0.9934
# 10000/10000 [==============================] - 1s 136us/step
# --------------------
# Accuracy on test data:  0.9974

# Get the digit from the probability of softmax layer
raw_pred = model.predict(test_scaled)

# pred = [ np.argmax(raw_pred[i]) for i in range(raw_pred.shape[0]) ]
pred = np.argmax(results, axis=1)

predictions = np.array(pred)

sample = pd.read_csv('data/sample_submission.csv')
result = pd.DataFrame({'ImageID': sample.ImageId, 'Label': predictions})
result.to_csv('cnn-submission-1', index=False)
print(result.head())
