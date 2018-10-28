"""
Kaggle Competition: Digit Recognizer
https://www.kaggle.com/c/digit-recognizer/data

Data: gray-scale images of hand-drawn digits, from zero through nine
https://www.kaggle.com/c/digit-recognizer/data

Algorithms Used: KNN, PCA
Submissions and Public Score:
1-KNN+PCA - 0.97185
1-KNN+PCA+4XData - 0.97214

References:
- https://www.kaggle.com/gregnetols/mnist-with-pca-and-knn
"""

import matplotlib
import numpy as np
import os
import pandas as pd
import pickle

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# Workaround for MacOS/conda setup
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def close_plot():
    plt.clf()
    plt.cla()
    plt.close()

def visualize(result):
    plt.scatter(result[:1000, 0], result[:1000, 1], c=y_train[:1000], cmap=plt.get_cmap('Pastel1', 10), s=5)
    plt.colorbar()
    plt.savefig('scatter-plot-aug.png')
    close_plot()

def plot_cumulative_variance(result):
    """Plot the cumulative sum of explained variances
    and find a reasonable cutoff
    """
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlabel('# of components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig('pca-cumulative-aug.png')
    close_plot()

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

y_train = train['label']
X_train = train.drop('label', axis=1)
X_submission = test

# pca_2comp = PCA(n_components=2).fit_transform(X_train)
# visualize(pca_2comp)

# pca_full = PCA(n_components=200).fit(X_train)
# plot_cumulative_variance(pca_full)

# Cutoff based on pca_full plot
pca = PCA(n_components=55)
X_train_transformed = pca.fit_transform(X_train)
X_submission_transformed = pca.transform(X_submission)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_train_transformed, y_train, test_size=0.2, random_state=42)

components = [15, 20, 25, 30, 35, 40, 45]
neighbors = [3, 4, 5, 6, 7]

# Create score 2D array
scores = np.zeros((components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ))

# Train the classifier and decide best component/neighbor combo
for component in components:
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train_pca[:, :component], y_train_pca)
        score = knn.score(X_test_pca[:, :component], y_test_pca)
        scores[component][n] = score

        print('Components = ', component, ', neighbors = ', n,', Score = ', score)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca[:, :35], y_train_pca)

predict_labels = knn.predict(X_submission_transformed[:, :35])

submission = pd.DataFrame({'ImageId': range(1, len(predict_labels) + 1), 'Label': predict_labels})

# Score starts at 0.94/15 and peaks at 0.973/30
# Submission 1 uses 35 components and 5 neighbors. Kaggle Score 0.97185
# submission.to_csv("submission-knn1.csv", index=False)

# Second submission uses train-augmented.csv from mnist-data-augmentation.py
# Score starts at 0.95/15 and peaks at 0.985/45
# Submission 2 uses 45 components and 3 neighbors. Kaggle Score 0.97214
submission.to_csv("submission-knn2.csv", index=False)

# Submission 3 uses 45 components and 7 neighbors. Kaggle Score 0.97100
