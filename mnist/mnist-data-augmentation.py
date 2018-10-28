"""
Kaggle Competition: Digit Recognizer
https://www.kaggle.com/c/digit-recognizer/data

Augment the training set by creating blurred, faded, zapped or noisy images
"""

import random
import numpy as np
import os
import pandas as pd

train = pd.read_csv('data/train.csv')

def blur(pixel):
    return pixel * random.uniform(0.8, 1.2)

def fade(pixel):
    return pixel * random.uniform(0.4, .8)

def zap(df):
    cols = random.sample(df.columns.tolist(), 156)
    for col in cols:
        df[col] = 0
    return df

def noise(df):
    cols = random.sample(df.columns.tolist(), 39)
    for col in cols:
        df[col] = 255 * random.uniform(0.1, 1)
    return df


y_train = train['label']
X_train = train.drop(['label'], axis=1)
X_train_blur = X_train.apply(blur)
X_train_zap = zap(X_train)
X_train_noise = noise(X_train)

print('generating images')
train_base = pd.concat([y_train, X_train], axis=1)
train_blur = pd.concat([y_train, X_train_blur], axis=1)
train_zap = pd.concat([y_train, X_train_zap], axis=1)
train_noise = pd.concat([y_train, X_train_noise], axis=1)

train_augmented = pd.concat([train_base, train_blur, train_zap, train_noise])
print('saving')

train_augmented.to_csv('train-augmented.csv', index=False)
