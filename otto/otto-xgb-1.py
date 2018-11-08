"""
Otto Group Product Classification Challenge
https://www.kaggle.com/c/otto-group-product-classification-challenge

Goal: distinguish between 10 main product categories
Data: dataset with 93 features for more than 200,000 products
https://www.kaggle.com/c/otto-group-product-classification-challenge/data

Algorithms Used: XGB
Submissions and Public Score:
1-XGB - 0.47791


Reference:
- https://www.kaggle.com/gaoyuan19930220/xgboost/code
"""

import pandas as pd
import os
import xgboost as xgb
import operator
from sklearn import preprocessing

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
sample = pd.read_csv('data/sampleSubmission.csv')

labels = train.target.values
labels = preprocessing.LabelEncoder().fit_transform(labels)
train = train.drop(['id', 'target'], axis=1)
features = list(train.columns[0:])
test = test.drop('id', axis=1)

# Model 1: XGBClassifier
params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}
train_xgb = xgb.DMatrix(train, labels)
test_xgb = xgb.DMatrix(test)
training_rounds = 100

print('Training...')
gbm = xgb.train(params, train_xgb, training_rounds)
pred = gbm.predict(test_xgb)

submission = pd.DataFrame(pred, index=sample.id.values, columns=sample.columns[1:])
submission.to_csv('xgb-1.csv', index_label='id')

