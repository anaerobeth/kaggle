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
