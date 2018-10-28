"""
Kaggle Competition: Digit Recognizer
https://www.kaggle.com/c/digit-recognizer/data

Data: gray-scale images of hand-drawn digits, from zero through nine
https://www.kaggle.com/c/digit-recognizer/data

Algorithms Used: KNN, PCA - Version 2
Submissions and Public Score:
1-KNN+PCA+4XData-50components - 0.97185
2-KNN+PCA+4XData-40components - 0.97154

Reference:
- https://www.kaggle.com/sflender/comparing-random-forest-pca-and-knn
"""

import matplotlib
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Workaround for MacOS/conda setup
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def evaluate_classifier(clf, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    clf.fit(trainX, trainY)
    return clf.score(testX,testY)

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

target = train['label']
train = train.drop('label', axis=1)

components_array = [1,2,3,4,5,10,20,50,100,200,500]

variance_ratio = np.zeros(len(components_array))
i = 0

print('PCA')
for components in components_array:
    pca = PCA(n_components=components)
    pca.fit(train)
    variance_ratio[i] = sum(pca.explained_variance_ratio_)
    i = i + 1

plt.plot(components_array, variance_ratio, 'k.-')
plt.xscale("log")
plt.ylim(9e-2,1.1)
plt.yticks(linspace(0.2,1.0,9))
plt.xlim(0.9)
plt.grid(which="both")
plt.xlabel("number of PCA components")
plt.ylabel("variance ratio")
plt.savefig('pca-variance-ratio.png')
plt.clf()
plt.cla()
plt.close()

print('training')
clf = KNeighborsClassifier()
scores = np.zeros(len(components_array))
i = 0

for components in components_array:
    pca = PCA(n_components=components)
    pca.fit(train)
    transform = pca.transform(train.iloc[0:1000])
    scores[i] = evaluate_classifier(clf, transform, target.iloc[0:1000], 0.8)
    i = i + 1

plt.plot(components_array, variance_ratio, 'k.-')
plt.xscale("log")
plt.grid(which="both")
plt.xlabel("number of PCA components")
plt.ylabel("accuracy")
plt.savefig('pca-accuracy.png')

print('predicting')

pca = PCA(n_components=40)
pca.fit(train)
transform_train = pca.transform(train)
transform_test = pca.transform(test)

clf = KNeighborsClassifier()
clf.fit(transform_train, target)
results=clf.predict(transform_test)

np.savetxt('submission-knn-pcaB-1.csv',
           np.c_[range(1,len(test)+1),results],
           delimiter=',',
           header = 'ImageId,Label',
           comments = '',
           fmt='%d')

