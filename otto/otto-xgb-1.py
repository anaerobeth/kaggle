"""
Otto Group Product Classification Challenge
https://www.kaggle.com/c/otto-group-product-classification-challenge

Goal: distinguish between 10 main product categories
Data: dataset with 93 features for more than 200,000 products
https://www.kaggle.com/c/otto-group-product-classification-challenge/data

Algorithms Used: XGB
Submissions and Public Score:
1-XGB - 0.47791
2-XGB+params+n_estimators=250 - 0.64972
3-XGB1+n_estimators=250 - 0.47791
4-XGB+scaled train- 1.35034
5-XGB+scaled train and test- 0.49806


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

# Model 2: XGB with n_estimator=250 and params from otto-xgb-gridsearch.py
params2 = {'base_score': 0.5, 'booster': 'gbtree', 'colsample_bylevel': 1, 'colsample_bytree': 1, 'gamma': 0, 'learning_rate': 0.1, 'max_delta_step': 0, 'max_depth': 3, 'min_child_weight': 1, 'n_estimators': 250, 'n_jobs': 1, 'objective': 'multi:softprob', 'random_state': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'scale_pos_weight': 1, 'seed': 42, 'silent': True, 'subsample': 1, 'num_class': 9}
gbm2 = xgb.train(params2, train_xgb, training_rounds)
pred2 = gbm2.predict(test_xgb)
submission2 = pd.DataFrame(pred2, index=sample.id.values, columns=sample.columns[1:])
submission2.to_csv('xgb-2.csv', index_label='id')

# Model 3: XGB with n_estimator=250 and default params
params3 = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9, 'n_estimator': 250}
gbm3 = xgb.train(params3, train_xgb, training_rounds)
pred3 = gbm3.predict(test_xgb)
submission3 = pd.DataFrame(pred3, index=sample.id.values, columns=sample.columns[1:])
submission3.to_csv('xgb-3.csv', index_label='id')


# Model 4: Use scaling and train_test_split

# Scale and split into test and dev
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(train), columns=train.columns)
X_test = pd.DataFrame(scaler.fit_transform(test), columns=test.columns)
dev_X, val_X, dev_y, val_y = train_test_split(X_train, labels, test_size = 0.2, random_state = 42)
params = {"objective": "multi:softprob", "eval_metric":"mlogloss", "num_class": 9}
training_rounds = 100
dev_xgb = xgb.DMatrix(dev_X, dev_y)
val_xgb = xgb.DMatrix(val_X)
gbm = xgb.train(params, dev_xgb, training_rounds)
pred_val = gbm.predict(val_xgb)

# Evaluate logloss
def logloss(num_observations, num_labels, labels, pred):
    total = 0
    for i in range(1, num_observations):
        for j in range(1, num_labels):
            yij = 1 if labels[i] == j else 0
            pij = pred[i][j]
            total += yij * np.log(pij)
    return -(1.0 / num_observations) * total

print(logloss(len(val_y), 9, val_y, pred_val))
# 0.4425

gbm.save_model('otto-xgb-4.model')
gbm.dump_model('otto-xgb-4.raw.txt')

pred4 = gbm.predict(X_test)
submission4 = pd.DataFrame(pred4, index=sample.id.values, columns=sample.columns[1:])
submission4.to_csv('xgb-4.csv', index_label='id')

X_test_xgb = xgb.DMatrix(X_test)
pred5 = gbm.predict(X_test_xgb)
submission5 = pd.DataFrame(pred5, index=sample.id.values, columns=sample.columns[1:])
submission5.to_csv('xgb-5.csv', index_label='id')

